from torchrl.collectors import SyncDataCollector
from tensordict import TensorDictBase
from torchrl.envs import set_exploration_type


class DifferentiableSyncDataCollector(SyncDataCollector):
    def __init__(*args, **kwargs):
        kwargs["use_buffers"] = False
        super().__init__(*args, **kwargs)

    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._shuttle.update(self.env.reset())

        # self._shuttle.fill_(("collector", "step_count"), 0)
        if self._use_buffers:
            self._final_rollout.fill_(("collector", "traj_ids"), -1)
        else:
            pass
        tensordicts = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if self.init_random_frames is not None and self._frames < self.init_random_frames:
                    self.env.rand_action(self._shuttle)
                    if self.policy_device is not None and self.policy_device != self.env_device:
                        # TODO: This may break with exclusive / ragged lazy stacks
                        self._shuttle.apply(
                            lambda name, val: val.to(device=self.policy_device, non_blocking=True) if name in self._policy_output_keys else val,
                            out=self._shuttle,
                            named=True,
                            nested_keys=True,
                        )
                else:
                    if self._cast_to_policy_device:
                        if self.policy_device is not None:
                            # This is unsafe if the shuttle is in pin_memory -- otherwise cuda will be happy with non_blocking
                            non_blocking = not self.no_cuda_sync or self.policy_device.type == "cuda"
                            policy_input = self._shuttle.to(
                                self.policy_device,
                                non_blocking=non_blocking,
                            )
                            if not self.no_cuda_sync:
                                self._sync_policy()
                        elif self.policy_device is None:
                            # we know the tensordict has a device otherwise we would not be here
                            # we can pass this, clear_device_ must have been called earlier
                            # policy_input = self._shuttle.clear_device_()
                            policy_input = self._shuttle
                    else:
                        policy_input = self._shuttle
                    # we still do the assignment for security
                    if self.compiled_policy:
                        cudagraph_mark_step_begin()
                    policy_output = self.policy(policy_input)
                    if self.compiled_policy:
                        policy_output = policy_output.clone()
                    if self._shuttle is not policy_output:
                        # ad-hoc update shuttle
                        self._shuttle.update(policy_output, keys_to_update=self._policy_output_keys)

                if self._cast_to_env_device:
                    if self.env_device is not None:
                        non_blocking = not self.no_cuda_sync or self.env_device.type == "cuda"
                        env_input = self._shuttle.to(self.env_device, non_blocking=non_blocking)
                        if not self.no_cuda_sync:
                            self._sync_env()
                    elif self.env_device is None:
                        # we know the tensordict has a device otherwise we would not be here
                        # we can pass this, clear_device_ must have been called earlier
                        # env_input = self._shuttle.clear_device_()
                        env_input = self._shuttle
                else:
                    env_input = self._shuttle
                env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

                if self._shuttle is not env_output:
                    # ad-hoc update shuttle
                    next_data = env_output.get("next")
                    if self._shuttle_has_no_device:
                        # Make sure
                        next_data.clear_device_()
                    self._shuttle.set("next", next_data)

                if self.replay_buffer is not None:
                    self.replay_buffer.add(self._shuttle)
                    if self._increment_frames(self._shuttle.numel()):
                        return
                else:
                    if self.storing_device is not None:
                        non_blocking = not self.no_cuda_sync or self.storing_device.type == "cuda"
                        tensordicts.append(self._shuttle.to(self.storing_device, non_blocking=non_blocking))
                        if not self.no_cuda_sync:
                            self._sync_storage()
                    else:
                        tensordicts.append(self._shuttle)

                # carry over collector data without messing up devices
                collector_data = self._shuttle.get("collector").copy()
                self._shuttle = env_next_output
                if self._shuttle_has_no_device:
                    self._shuttle.clear_device_()
                self._shuttle.set("collector", collector_data)
                self._update_traj_ids(env_output)

                if self.interruptor is not None and self.interruptor.collection_stopped():
                    if self.replay_buffer is not None:
                        return
                    result = self._final_rollout
                    if self._use_buffers:
                        try:
                            torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout[..., : t + 1],
                            )
                        except RuntimeError:
                            with self._final_rollout.unlock_():
                                torch.stack(
                                    tensordicts,
                                    self._final_rollout.ndim - 1,
                                    out=self._final_rollout[..., : t + 1],
                                )
                    else:
                        result = TensorDict.maybe_dense_stack(tensordicts, dim=-1)
                    break
            else:
                if self._use_buffers:
                    result = self._final_rollout
                    try:
                        result = torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout,
                        )

                    except RuntimeError:
                        with self._final_rollout.unlock_():
                            result = torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout,
                            )
                elif self.replay_buffer is not None:
                    return
                else:
                    result = TensorDict.maybe_dense_stack(tensordicts, dim=-1)
                    result.refine_names(..., "time")

        return self._maybe_set_truncated(result)

    def reset(self, index=None, **kwargs) -> None:
        """Resets the environments to a new initial state."""
        # metadata
        collector_metadata = self._shuttle.get("collector").clone()
        if index is not None:
            # check that the env supports partial reset
            if prod(self.env.batch_size) == 0:
                raise RuntimeError("resetting unique env with index is not permitted.")
            for reset_key, done_keys in zip(self.env.reset_keys, self.env.done_keys_groups):
                _reset = torch.zeros(
                    self.env.full_done_spec[done_keys[0]].shape,
                    dtype=torch.bool,
                    device=self.env.device,
                )
                _reset[index] = 1
                self._shuttle.set(reset_key, _reset)
        else:
            _reset = None
            self._shuttle.zero_()

        self._shuttle.update(self.env.reset(**kwargs), inplace=True)
        collector_metadata["traj_ids"] = collector_metadata["traj_ids"] - collector_metadata["traj_ids"].min()
        self._shuttle["collector"] = collector_metadata
