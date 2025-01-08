
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.max_autotune = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.graph_diagram = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.1
# torch cuda version: None
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor([   0.0000,    0.0000, -665.1180]))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1):
        full_default = torch.ops.aten.full.default([16, 1023, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False);  full_default = None
        select = torch.ops.aten.select.int(arg0_1, 1, 0)
        view = torch.ops.aten.view.default(select, [16, 1]);  select = None
        cos = torch.ops.aten.cos.default(view)
        sin = torch.ops.aten.sin.default(view);  view = None
        full_default_1 = torch.ops.aten.full.default([16, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([16, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        cat = torch.ops.aten.cat.default([cos, full_default_1, sin], -1)
        cat_1 = torch.ops.aten.cat.default([full_default_1, full_default_2, full_default_1], -1);  full_default_2 = None
        neg = torch.ops.aten.neg.default(sin);  sin = None
        cat_2 = torch.ops.aten.cat.default([neg, full_default_1, cos], -1);  neg = full_default_1 = cos = None
        cat_3 = torch.ops.aten.cat.default([cat, cat_1, cat_2], 1);  cat = cat_1 = cat_2 = None
        view_1 = torch.ops.aten.view.default(cat_3, [16, 3, 3]);  cat_3 = None
        select_1 = torch.ops.aten.select.int(arg2_1, 0, 0)
        sub = torch.ops.aten.sub.Tensor(arg1_1, select_1)
        bmm = torch.ops.aten.bmm.default(sub, view_1);  sub = view_1 = None
        add = torch.ops.aten.add.Tensor(bmm, select_1);  bmm = select_1 = None
        select_2 = torch.ops.aten.select.int(arg3_1, 0, 0)
        unsqueeze = torch.ops.aten.unsqueeze.default(select_2, -1);  select_2 = None
        mul = torch.ops.aten.mul.Tensor(unsqueeze, add);  unsqueeze = add = None
        select_3 = torch.ops.aten.select.int(arg0_1, 1, 1)
        view_2 = torch.ops.aten.view.default(select_3, [16, 1]);  select_3 = None
        cos_1 = torch.ops.aten.cos.default(view_2)
        sin_1 = torch.ops.aten.sin.default(view_2);  view_2 = None
        full_default_3 = torch.ops.aten.full.default([16, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        full_default_4 = torch.ops.aten.full.default([16, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        cat_4 = torch.ops.aten.cat.default([cos_1, full_default_3, sin_1], -1)
        cat_5 = torch.ops.aten.cat.default([full_default_3, full_default_4, full_default_3], -1);  full_default_4 = None
        neg_1 = torch.ops.aten.neg.default(sin_1);  sin_1 = None
        cat_6 = torch.ops.aten.cat.default([neg_1, full_default_3, cos_1], -1);  neg_1 = full_default_3 = cos_1 = None
        cat_7 = torch.ops.aten.cat.default([cat_4, cat_5, cat_6], 1);  cat_4 = cat_5 = cat_6 = None
        view_3 = torch.ops.aten.view.default(cat_7, [16, 3, 3]);  cat_7 = None
        select_4 = torch.ops.aten.select.int(arg2_1, 0, 1)
        sub_1 = torch.ops.aten.sub.Tensor(arg1_1, select_4)
        bmm_1 = torch.ops.aten.bmm.default(sub_1, view_3);  sub_1 = view_3 = None
        add_2 = torch.ops.aten.add.Tensor(bmm_1, select_4);  bmm_1 = select_4 = None
        select_5 = torch.ops.aten.select.int(arg3_1, 0, 1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(select_5, -1);  select_5 = None
        mul_1 = torch.ops.aten.mul.Tensor(unsqueeze_1, add_2);  unsqueeze_1 = add_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        select_6 = torch.ops.aten.select.int(arg0_1, 1, 2)
        view_4 = torch.ops.aten.view.default(select_6, [16, 1]);  select_6 = None
        cos_2 = torch.ops.aten.cos.default(view_4)
        sin_2 = torch.ops.aten.sin.default(view_4);  view_4 = None
        full_default_5 = torch.ops.aten.full.default([16, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        full_default_6 = torch.ops.aten.full.default([16, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        cat_8 = torch.ops.aten.cat.default([cos_2, full_default_5, sin_2], -1)
        cat_9 = torch.ops.aten.cat.default([full_default_5, full_default_6, full_default_5], -1);  full_default_6 = None
        neg_2 = torch.ops.aten.neg.default(sin_2);  sin_2 = None
        cat_10 = torch.ops.aten.cat.default([neg_2, full_default_5, cos_2], -1);  neg_2 = full_default_5 = cos_2 = None
        cat_11 = torch.ops.aten.cat.default([cat_8, cat_9, cat_10], 1);  cat_8 = cat_9 = cat_10 = None
        view_5 = torch.ops.aten.view.default(cat_11, [16, 3, 3]);  cat_11 = None
        select_7 = torch.ops.aten.select.int(arg2_1, 0, 2)
        sub_2 = torch.ops.aten.sub.Tensor(arg1_1, select_7)
        bmm_2 = torch.ops.aten.bmm.default(sub_2, view_5);  sub_2 = view_5 = None
        add_4 = torch.ops.aten.add.Tensor(bmm_2, select_7);  bmm_2 = select_7 = None
        select_8 = torch.ops.aten.select.int(arg3_1, 0, 2)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(select_8, -1);  select_8 = None
        mul_2 = torch.ops.aten.mul.Tensor(unsqueeze_2, add_4);  unsqueeze_2 = add_4 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, mul_2);  add_3 = mul_2 = None
        select_9 = torch.ops.aten.select.int(arg0_1, 1, 3)
        view_6 = torch.ops.aten.view.default(select_9, [16, 1]);  select_9 = None
        cos_3 = torch.ops.aten.cos.default(view_6)
        sin_3 = torch.ops.aten.sin.default(view_6);  view_6 = None
        full_default_7 = torch.ops.aten.full.default([16, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        full_default_8 = torch.ops.aten.full.default([16, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        cat_12 = torch.ops.aten.cat.default([cos_3, full_default_7, sin_3], -1)
        cat_13 = torch.ops.aten.cat.default([full_default_7, full_default_8, full_default_7], -1);  full_default_8 = None
        neg_3 = torch.ops.aten.neg.default(sin_3);  sin_3 = None
        cat_14 = torch.ops.aten.cat.default([neg_3, full_default_7, cos_3], -1);  neg_3 = full_default_7 = cos_3 = None
        cat_15 = torch.ops.aten.cat.default([cat_12, cat_13, cat_14], 1);  cat_12 = cat_13 = cat_14 = None
        view_7 = torch.ops.aten.view.default(cat_15, [16, 3, 3]);  cat_15 = None
        select_10 = torch.ops.aten.select.int(arg2_1, 0, 3);  arg2_1 = None
        sub_3 = torch.ops.aten.sub.Tensor(arg1_1, select_10)
        bmm_3 = torch.ops.aten.bmm.default(sub_3, view_7);  sub_3 = view_7 = None
        add_6 = torch.ops.aten.add.Tensor(bmm_3, select_10);  bmm_3 = select_10 = None
        select_11 = torch.ops.aten.select.int(arg3_1, 0, 3)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(select_11, -1);  select_11 = None
        mul_3 = torch.ops.aten.mul.Tensor(unsqueeze_3, add_6);  unsqueeze_3 = add_6 = None
        add_7 = torch.ops.aten.add.Tensor(add_5, mul_3);  add_5 = mul_3 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(unsqueeze_4, arg1_1);  unsqueeze_4 = arg1_1 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, mul_4);  add_7 = mul_4 = None
        view_8 = torch.ops.aten.view.default(arg5_1, [16, 1, 3])
        permute = torch.ops.aten.permute.default(arg6_1, [0, 2, 1])
        bmm_4 = torch.ops.aten.bmm.default(add_8, permute);  permute = None
        add_9 = torch.ops.aten.add.Tensor(bmm_4, view_8);  bmm_4 = view_8 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(arg7_1, 1)
        mul_5 = torch.ops.aten.mul.Tensor(unsqueeze_5, add_9);  unsqueeze_5 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_5, [-2]);  mul_5 = None
        sum_2 = torch.ops.aten.sum.default(arg7_1)
        div = torch.ops.aten.div.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(div, 1)
        sub_4 = torch.ops.aten.sub.Tensor(add_9, unsqueeze_6);  unsqueeze_6 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_4, sub_4)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(arg7_1, 1)
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, unsqueeze_7);  mul_6 = unsqueeze_7 = None
        select_12 = torch.ops.aten.select.int(sub_4, 2, 0)
        select_13 = torch.ops.aten.select.int(sub_4, 2, 1)
        select_14 = torch.ops.aten.select.int(sub_4, 2, 2)
        select_15 = torch.ops.aten.select.int(mul_7, 2, 0)
        select_16 = torch.ops.aten.select.int(mul_7, 2, 1)
        select_17 = torch.ops.aten.select.int(mul_7, 2, 2);  mul_7 = None
        add_10 = torch.ops.aten.add.Tensor(select_16, select_17)
        sum_3 = torch.ops.aten.sum.dim_IntList(add_10, [-1]);  add_10 = None
        add_11 = torch.ops.aten.add.Tensor(select_15, select_17);  select_17 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(add_11, [-1]);  add_11 = None
        add_12 = torch.ops.aten.add.Tensor(select_15, select_16);  select_15 = select_16 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(add_12, [-1]);  add_12 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg7_1, 0)
        mul_8 = torch.ops.aten.mul.Tensor(unsqueeze_8, select_12);  unsqueeze_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, select_13);  mul_8 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_9, [-1]);  mul_9 = None
        neg_4 = torch.ops.aten.neg.default(sum_6);  sum_6 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(arg7_1, 0)
        mul_10 = torch.ops.aten.mul.Tensor(unsqueeze_9, select_12);  unsqueeze_9 = select_12 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, select_14);  mul_10 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_11, [-1]);  mul_11 = None
        neg_5 = torch.ops.aten.neg.default(sum_7);  sum_7 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(arg7_1, 0);  arg7_1 = None
        mul_12 = torch.ops.aten.mul.Tensor(unsqueeze_10, select_13);  unsqueeze_10 = select_13 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, select_14);  mul_12 = select_14 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_13, [-1]);  mul_13 = None
        neg_6 = torch.ops.aten.neg.default(sum_8);  sum_8 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(sum_3, 1);  sum_3 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(neg_4, 1)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(neg_5, 1)
        cat_16 = torch.ops.aten.cat.default([unsqueeze_11, unsqueeze_12, unsqueeze_13], -1);  unsqueeze_11 = unsqueeze_12 = unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(neg_4, 1);  neg_4 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(sum_4, 1);  sum_4 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(neg_6, 1)
        cat_17 = torch.ops.aten.cat.default([unsqueeze_14, unsqueeze_15, unsqueeze_16], -1);  unsqueeze_14 = unsqueeze_15 = unsqueeze_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(neg_5, 1);  neg_5 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(neg_6, 1);  neg_6 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(sum_5, 1);  sum_5 = None
        cat_18 = torch.ops.aten.cat.default([unsqueeze_17, unsqueeze_18, unsqueeze_19], -1);  unsqueeze_17 = unsqueeze_18 = unsqueeze_19 = None
        cat_19 = torch.ops.aten.cat.default([cat_16, cat_17, cat_18], 1);  cat_16 = cat_17 = cat_18 = None
        view_10 = torch.ops.aten.view.default(cat_19, [16, 3, 3]);  cat_19 = None
        slice_10 = torch.ops.aten.slice.Tensor(add_9, 2, 0, 2)
        div_1 = torch.ops.aten.div.Tensor(slice_10, 6.4);  slice_10 = None
        clamp_min = torch.ops.aten.clamp_min.default(div_1, -1);  div_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 1);  clamp_min = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(clamp_max, 2);  clamp_max = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None
        iota = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_11 = torch.ops.aten.view.default(iota, [16, 1, 1, 1]);  iota = None
        iota_1 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_12 = torch.ops.aten.view.default(iota_1, [1, 1, 1, 1]);  iota_1 = None
        select_18 = torch.ops.aten.select.int(unsqueeze_20, 3, 0)
        select_19 = torch.ops.aten.select.int(unsqueeze_20, 3, 1);  unsqueeze_20 = None
        mul_14 = torch.ops.aten.mul.Tensor(select_18, 127.5);  select_18 = None
        add_13 = torch.ops.aten.add.Tensor(mul_14, 127.5);  mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(select_19, 127.5);  select_19 = None
        add_14 = torch.ops.aten.add.Tensor(mul_15, 127.5);  mul_15 = None
        floor = torch.ops.aten.floor.default(add_13)
        floor_1 = torch.ops.aten.floor.default(add_14)
        add_15 = torch.ops.aten.add.Tensor(floor, 1)
        add_16 = torch.ops.aten.add.Tensor(floor_1, 1)
        sub_5 = torch.ops.aten.sub.Tensor(add_15, add_13)
        sub_6 = torch.ops.aten.sub.Tensor(add_16, add_14)
        mul_16 = torch.ops.aten.mul.Tensor(sub_5, sub_6);  sub_5 = sub_6 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_13, floor)
        sub_8 = torch.ops.aten.sub.Tensor(add_16, add_14)
        mul_17 = torch.ops.aten.mul.Tensor(sub_7, sub_8);  sub_7 = sub_8 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_15, add_13)
        sub_10 = torch.ops.aten.sub.Tensor(add_14, floor_1)
        mul_18 = torch.ops.aten.mul.Tensor(sub_9, sub_10);  sub_9 = sub_10 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_13, floor);  add_13 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_14, floor_1);  add_14 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_11, sub_12);  sub_11 = sub_12 = None
        ge = torch.ops.aten.ge.Scalar(floor, 0)
        lt = torch.ops.aten.lt.Scalar(floor, 256)
        ge_1 = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_1 = torch.ops.aten.lt.Scalar(floor_1, 256)
        logical_and = torch.ops.aten.logical_and.default(ge_1, lt_1);  ge_1 = lt_1 = None
        logical_and_1 = torch.ops.aten.logical_and.default(lt, logical_and);  lt = logical_and = None
        logical_and_2 = torch.ops.aten.logical_and.default(ge, logical_and_1);  ge = logical_and_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(floor_1, torch.int64)
        full_default_9 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(logical_and_2, convert_element_type, full_default_9);  convert_element_type = full_default_9 = None
        view_13 = torch.ops.aten.view.default(where, [16, 1, 1023, 1]);  where = None
        full_default_10 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_1 = torch.ops.aten.where.self(logical_and_2, convert_element_type_1, full_default_10);  convert_element_type_1 = full_default_10 = None
        view_14 = torch.ops.aten.view.default(where_1, [16, 1, 1023, 1]);  where_1 = None
        full_default_11 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_2 = torch.ops.aten.where.self(logical_and_2, mul_16, full_default_11);  logical_and_2 = mul_16 = full_default_11 = None
        view_15 = torch.ops.aten.view.default(where_2, [16, 1, 1023, 1]);  where_2 = None
        index = torch.ops.aten.index.Tensor(unsqueeze_21, [view_11, view_12, view_14, view_13]);  view_14 = view_13 = None
        mul_20 = torch.ops.aten.mul.Tensor(index, view_15);  index = view_15 = None
        ge_2 = torch.ops.aten.ge.Scalar(add_15, 0)
        lt_2 = torch.ops.aten.lt.Scalar(add_15, 256)
        ge_3 = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_3 = torch.ops.aten.lt.Scalar(floor_1, 256)
        logical_and_3 = torch.ops.aten.logical_and.default(ge_3, lt_3);  ge_3 = lt_3 = None
        logical_and_4 = torch.ops.aten.logical_and.default(lt_2, logical_and_3);  lt_2 = logical_and_3 = None
        logical_and_5 = torch.ops.aten.logical_and.default(ge_2, logical_and_4);  ge_2 = logical_and_4 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_15, torch.int64)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(floor_1, torch.int64);  floor_1 = None
        full_default_12 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_3 = torch.ops.aten.where.self(logical_and_5, convert_element_type_2, full_default_12);  convert_element_type_2 = full_default_12 = None
        view_16 = torch.ops.aten.view.default(where_3, [16, 1, 1023, 1]);  where_3 = None
        full_default_13 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_4 = torch.ops.aten.where.self(logical_and_5, convert_element_type_3, full_default_13);  convert_element_type_3 = full_default_13 = None
        view_17 = torch.ops.aten.view.default(where_4, [16, 1, 1023, 1]);  where_4 = None
        full_default_14 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_5 = torch.ops.aten.where.self(logical_and_5, mul_17, full_default_14);  logical_and_5 = mul_17 = full_default_14 = None
        view_18 = torch.ops.aten.view.default(where_5, [16, 1, 1023, 1]);  where_5 = None
        index_1 = torch.ops.aten.index.Tensor(unsqueeze_21, [view_11, view_12, view_17, view_16]);  view_17 = view_16 = None
        mul_21 = torch.ops.aten.mul.Tensor(index_1, view_18);  index_1 = view_18 = None
        add_17 = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
        ge_4 = torch.ops.aten.ge.Scalar(floor, 0)
        lt_4 = torch.ops.aten.lt.Scalar(floor, 256)
        ge_5 = torch.ops.aten.ge.Scalar(add_16, 0)
        lt_5 = torch.ops.aten.lt.Scalar(add_16, 256)
        logical_and_6 = torch.ops.aten.logical_and.default(ge_5, lt_5);  ge_5 = lt_5 = None
        logical_and_7 = torch.ops.aten.logical_and.default(lt_4, logical_and_6);  lt_4 = logical_and_6 = None
        logical_and_8 = torch.ops.aten.logical_and.default(ge_4, logical_and_7);  ge_4 = logical_and_7 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(floor, torch.int64);  floor = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(add_16, torch.int64)
        full_default_15 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_6 = torch.ops.aten.where.self(logical_and_8, convert_element_type_4, full_default_15);  convert_element_type_4 = full_default_15 = None
        view_19 = torch.ops.aten.view.default(where_6, [16, 1, 1023, 1]);  where_6 = None
        full_default_16 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_7 = torch.ops.aten.where.self(logical_and_8, convert_element_type_5, full_default_16);  convert_element_type_5 = full_default_16 = None
        view_20 = torch.ops.aten.view.default(where_7, [16, 1, 1023, 1]);  where_7 = None
        full_default_17 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_8 = torch.ops.aten.where.self(logical_and_8, mul_18, full_default_17);  logical_and_8 = mul_18 = full_default_17 = None
        view_21 = torch.ops.aten.view.default(where_8, [16, 1, 1023, 1]);  where_8 = None
        index_2 = torch.ops.aten.index.Tensor(unsqueeze_21, [view_11, view_12, view_20, view_19]);  view_20 = view_19 = None
        mul_22 = torch.ops.aten.mul.Tensor(index_2, view_21);  index_2 = view_21 = None
        add_18 = torch.ops.aten.add.Tensor(add_17, mul_22);  add_17 = mul_22 = None
        ge_6 = torch.ops.aten.ge.Scalar(add_15, 0)
        lt_6 = torch.ops.aten.lt.Scalar(add_15, 256)
        ge_7 = torch.ops.aten.ge.Scalar(add_16, 0)
        lt_7 = torch.ops.aten.lt.Scalar(add_16, 256)
        logical_and_9 = torch.ops.aten.logical_and.default(ge_7, lt_7);  ge_7 = lt_7 = None
        logical_and_10 = torch.ops.aten.logical_and.default(lt_6, logical_and_9);  lt_6 = logical_and_9 = None
        logical_and_11 = torch.ops.aten.logical_and.default(ge_6, logical_and_10);  ge_6 = logical_and_10 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(add_15, torch.int64);  add_15 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_16, torch.int64);  add_16 = None
        full_default_18 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_9 = torch.ops.aten.where.self(logical_and_11, convert_element_type_6, full_default_18);  convert_element_type_6 = full_default_18 = None
        view_22 = torch.ops.aten.view.default(where_9, [16, 1, 1023, 1]);  where_9 = None
        full_default_19 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_10 = torch.ops.aten.where.self(logical_and_11, convert_element_type_7, full_default_19);  convert_element_type_7 = full_default_19 = None
        view_23 = torch.ops.aten.view.default(where_10, [16, 1, 1023, 1]);  where_10 = None
        full_default_20 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_11 = torch.ops.aten.where.self(logical_and_11, mul_19, full_default_20);  logical_and_11 = mul_19 = full_default_20 = None
        view_24 = torch.ops.aten.view.default(where_11, [16, 1, 1023, 1]);  where_11 = None
        index_3 = torch.ops.aten.index.Tensor(unsqueeze_21, [view_11, view_12, view_23, view_22]);  unsqueeze_21 = view_11 = view_12 = view_23 = view_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(index_3, view_24);  index_3 = view_24 = None
        add_19 = torch.ops.aten.add.Tensor(add_18, mul_23);  add_18 = mul_23 = None
        squeeze = torch.ops.aten.squeeze.dim(add_19, 1);  add_19 = None
        slice_11 = torch.ops.aten.slice.Tensor(add_9, 2, 2, 3)
        sub_13 = torch.ops.aten.sub.Tensor(slice_11, squeeze);  slice_11 = squeeze = None
        slice_12 = torch.ops.aten.slice.Tensor(add_9, 2, 0, 2)
        clamp_min_1 = torch.ops.aten.clamp_min.default(slice_12, -6.4);  slice_12 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 6.4);  clamp_min_1 = None
        slice_13 = torch.ops.aten.slice.Tensor(add_9, 2, 0, 2)
        eq = torch.ops.aten.eq.Tensor(clamp_max_1, slice_13);  clamp_max_1 = slice_13 = None
        logical_not = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1 = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        le = torch.ops.aten.le.Scalar(sub_13, 0.0)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(le, logical_not_1);  le = logical_not_1 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(bitwise_and, torch.float32);  bitwise_and = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_13, convert_element_type_8);  sub_13 = None
        slice_14 = torch.ops.aten.slice.Tensor(add_9, 2, 0, 2)
        div_2 = torch.ops.aten.div.Tensor(slice_14, 6.4);  slice_14 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(div_2, -1);  div_2 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 1);  clamp_min_2 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(clamp_max_2, 2);  clamp_max_2 = None
        iota_2 = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_25 = torch.ops.aten.view.default(iota_2, [16, 1, 1, 1]);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_26 = torch.ops.aten.view.default(iota_3, [1, 2, 1, 1]);  iota_3 = None
        select_20 = torch.ops.aten.select.int(unsqueeze_22, 3, 0)
        select_21 = torch.ops.aten.select.int(unsqueeze_22, 3, 1);  unsqueeze_22 = None
        mul_25 = torch.ops.aten.mul.Tensor(select_20, 127.5);  select_20 = None
        add_20 = torch.ops.aten.add.Tensor(mul_25, 127.5);  mul_25 = None
        mul_26 = torch.ops.aten.mul.Tensor(select_21, 127.5);  select_21 = None
        add_21 = torch.ops.aten.add.Tensor(mul_26, 127.5);  mul_26 = None
        floor_2 = torch.ops.aten.floor.default(add_20)
        floor_3 = torch.ops.aten.floor.default(add_21)
        add_22 = torch.ops.aten.add.Tensor(floor_2, 1)
        add_23 = torch.ops.aten.add.Tensor(floor_3, 1)
        sub_14 = torch.ops.aten.sub.Tensor(add_22, add_20)
        sub_15 = torch.ops.aten.sub.Tensor(add_23, add_21)
        mul_27 = torch.ops.aten.mul.Tensor(sub_14, sub_15);  sub_14 = sub_15 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_20, floor_2)
        sub_17 = torch.ops.aten.sub.Tensor(add_23, add_21)
        mul_28 = torch.ops.aten.mul.Tensor(sub_16, sub_17);  sub_16 = sub_17 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_22, add_20)
        sub_19 = torch.ops.aten.sub.Tensor(add_21, floor_3)
        mul_29 = torch.ops.aten.mul.Tensor(sub_18, sub_19);  sub_18 = sub_19 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_20, floor_2);  add_20 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_21, floor_3);  add_21 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_20, sub_21);  sub_20 = sub_21 = None
        ge_8 = torch.ops.aten.ge.Scalar(floor_2, 0)
        lt_8 = torch.ops.aten.lt.Scalar(floor_2, 256)
        ge_9 = torch.ops.aten.ge.Scalar(floor_3, 0)
        lt_9 = torch.ops.aten.lt.Scalar(floor_3, 256)
        logical_and_12 = torch.ops.aten.logical_and.default(ge_9, lt_9);  ge_9 = lt_9 = None
        logical_and_13 = torch.ops.aten.logical_and.default(lt_8, logical_and_12);  lt_8 = logical_and_12 = None
        logical_and_14 = torch.ops.aten.logical_and.default(ge_8, logical_and_13);  ge_8 = logical_and_13 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(floor_2, torch.int64)
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(floor_3, torch.int64)
        full_default_21 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_12 = torch.ops.aten.where.self(logical_and_14, convert_element_type_9, full_default_21);  convert_element_type_9 = full_default_21 = None
        view_27 = torch.ops.aten.view.default(where_12, [16, 1, 1023, 1]);  where_12 = None
        full_default_22 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_13 = torch.ops.aten.where.self(logical_and_14, convert_element_type_10, full_default_22);  convert_element_type_10 = full_default_22 = None
        view_28 = torch.ops.aten.view.default(where_13, [16, 1, 1023, 1]);  where_13 = None
        full_default_23 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_14 = torch.ops.aten.where.self(logical_and_14, mul_27, full_default_23);  logical_and_14 = mul_27 = full_default_23 = None
        view_29 = torch.ops.aten.view.default(where_14, [16, 1, 1023, 1]);  where_14 = None
        index_4 = torch.ops.aten.index.Tensor(arg9_1, [view_25, view_26, view_28, view_27]);  view_28 = view_27 = None
        mul_31 = torch.ops.aten.mul.Tensor(index_4, view_29);  index_4 = view_29 = None
        ge_10 = torch.ops.aten.ge.Scalar(add_22, 0)
        lt_10 = torch.ops.aten.lt.Scalar(add_22, 256)
        ge_11 = torch.ops.aten.ge.Scalar(floor_3, 0)
        lt_11 = torch.ops.aten.lt.Scalar(floor_3, 256)
        logical_and_15 = torch.ops.aten.logical_and.default(ge_11, lt_11);  ge_11 = lt_11 = None
        logical_and_16 = torch.ops.aten.logical_and.default(lt_10, logical_and_15);  lt_10 = logical_and_15 = None
        logical_and_17 = torch.ops.aten.logical_and.default(ge_10, logical_and_16);  ge_10 = logical_and_16 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_22, torch.int64)
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(floor_3, torch.int64);  floor_3 = None
        full_default_24 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_15 = torch.ops.aten.where.self(logical_and_17, convert_element_type_11, full_default_24);  convert_element_type_11 = full_default_24 = None
        view_30 = torch.ops.aten.view.default(where_15, [16, 1, 1023, 1]);  where_15 = None
        full_default_25 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_16 = torch.ops.aten.where.self(logical_and_17, convert_element_type_12, full_default_25);  convert_element_type_12 = full_default_25 = None
        view_31 = torch.ops.aten.view.default(where_16, [16, 1, 1023, 1]);  where_16 = None
        full_default_26 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_17 = torch.ops.aten.where.self(logical_and_17, mul_28, full_default_26);  logical_and_17 = mul_28 = full_default_26 = None
        view_32 = torch.ops.aten.view.default(where_17, [16, 1, 1023, 1]);  where_17 = None
        index_5 = torch.ops.aten.index.Tensor(arg9_1, [view_25, view_26, view_31, view_30]);  view_31 = view_30 = None
        mul_32 = torch.ops.aten.mul.Tensor(index_5, view_32);  index_5 = view_32 = None
        add_24 = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
        ge_12 = torch.ops.aten.ge.Scalar(floor_2, 0)
        lt_12 = torch.ops.aten.lt.Scalar(floor_2, 256)
        ge_13 = torch.ops.aten.ge.Scalar(add_23, 0)
        lt_13 = torch.ops.aten.lt.Scalar(add_23, 256)
        logical_and_18 = torch.ops.aten.logical_and.default(ge_13, lt_13);  ge_13 = lt_13 = None
        logical_and_19 = torch.ops.aten.logical_and.default(lt_12, logical_and_18);  lt_12 = logical_and_18 = None
        logical_and_20 = torch.ops.aten.logical_and.default(ge_12, logical_and_19);  ge_12 = logical_and_19 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(floor_2, torch.int64);  floor_2 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_23, torch.int64)
        full_default_27 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_18 = torch.ops.aten.where.self(logical_and_20, convert_element_type_13, full_default_27);  convert_element_type_13 = full_default_27 = None
        view_33 = torch.ops.aten.view.default(where_18, [16, 1, 1023, 1]);  where_18 = None
        full_default_28 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_19 = torch.ops.aten.where.self(logical_and_20, convert_element_type_14, full_default_28);  convert_element_type_14 = full_default_28 = None
        view_34 = torch.ops.aten.view.default(where_19, [16, 1, 1023, 1]);  where_19 = None
        full_default_29 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_20 = torch.ops.aten.where.self(logical_and_20, mul_29, full_default_29);  logical_and_20 = mul_29 = full_default_29 = None
        view_35 = torch.ops.aten.view.default(where_20, [16, 1, 1023, 1]);  where_20 = None
        index_6 = torch.ops.aten.index.Tensor(arg9_1, [view_25, view_26, view_34, view_33]);  view_34 = view_33 = None
        mul_33 = torch.ops.aten.mul.Tensor(index_6, view_35);  index_6 = view_35 = None
        add_25 = torch.ops.aten.add.Tensor(add_24, mul_33);  add_24 = mul_33 = None
        ge_14 = torch.ops.aten.ge.Scalar(add_22, 0)
        lt_14 = torch.ops.aten.lt.Scalar(add_22, 256)
        ge_15 = torch.ops.aten.ge.Scalar(add_23, 0)
        lt_15 = torch.ops.aten.lt.Scalar(add_23, 256)
        logical_and_21 = torch.ops.aten.logical_and.default(ge_15, lt_15);  ge_15 = lt_15 = None
        logical_and_22 = torch.ops.aten.logical_and.default(lt_14, logical_and_21);  lt_14 = logical_and_21 = None
        logical_and_23 = torch.ops.aten.logical_and.default(ge_14, logical_and_22);  ge_14 = logical_and_22 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_22, torch.int64);  add_22 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(add_23, torch.int64);  add_23 = None
        full_default_30 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_21 = torch.ops.aten.where.self(logical_and_23, convert_element_type_15, full_default_30);  convert_element_type_15 = full_default_30 = None
        view_36 = torch.ops.aten.view.default(where_21, [16, 1, 1023, 1]);  where_21 = None
        full_default_31 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_22 = torch.ops.aten.where.self(logical_and_23, convert_element_type_16, full_default_31);  convert_element_type_16 = full_default_31 = None
        view_37 = torch.ops.aten.view.default(where_22, [16, 1, 1023, 1]);  where_22 = None
        full_default_32 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_23 = torch.ops.aten.where.self(logical_and_23, mul_30, full_default_32);  logical_and_23 = mul_30 = full_default_32 = None
        view_38 = torch.ops.aten.view.default(where_23, [16, 1, 1023, 1]);  where_23 = None
        index_7 = torch.ops.aten.index.Tensor(arg9_1, [view_25, view_26, view_37, view_36]);  arg9_1 = view_25 = view_26 = view_37 = view_36 = None
        mul_34 = torch.ops.aten.mul.Tensor(index_7, view_38);  index_7 = view_38 = None
        add_26 = torch.ops.aten.add.Tensor(add_25, mul_34);  add_25 = mul_34 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(add_26, -1);  add_26 = None
        permute_1 = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1]);  squeeze_1 = None
        neg_7 = torch.ops.aten.neg.default(permute_1);  permute_1 = None
        full_default_33 = torch.ops.aten.full.default([16, 1023, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        cat_20 = torch.ops.aten.cat.default([neg_7, full_default_33], 2);  neg_7 = full_default_33 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(cat_20, 2)
        sum_9 = torch.ops.aten.sum.dim_IntList(pow_1, [-1], True);  pow_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sum_9, 0.5);  sum_9 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(pow_2, 1e-06);  pow_2 = None
        div_3 = torch.ops.aten.div.Tensor(cat_20, clamp_min_3);  cat_20 = clamp_min_3 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(arg10_1, 1)
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(arg11_1, 1)
        expand = torch.ops.aten.expand.default(unsqueeze_24, [16, 1023, 3]);  unsqueeze_24 = None
        iota_4 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add_27 = torch.ops.aten.add.Tensor(iota_4, 1)
        remainder = torch.ops.aten.remainder.Scalar(add_27, 3);  add_27 = None
        index_8 = torch.ops.aten.index.Tensor(expand, [None, None, remainder]);  remainder = None
        add_28 = torch.ops.aten.add.Tensor(iota_4, 2)
        remainder_1 = torch.ops.aten.remainder.Scalar(add_28, 3);  add_28 = None
        index_9 = torch.ops.aten.index.Tensor(sub_4, [None, None, remainder_1]);  remainder_1 = None
        mul_35 = torch.ops.aten.mul.Tensor(index_8, index_9);  index_8 = index_9 = None
        add_29 = torch.ops.aten.add.Tensor(iota_4, 2)
        remainder_2 = torch.ops.aten.remainder.Scalar(add_29, 3);  add_29 = None
        index_10 = torch.ops.aten.index.Tensor(expand, [None, None, remainder_2]);  expand = remainder_2 = None
        add_30 = torch.ops.aten.add.Tensor(iota_4, 1);  iota_4 = None
        remainder_3 = torch.ops.aten.remainder.Scalar(add_30, 3);  add_30 = None
        index_11 = torch.ops.aten.index.Tensor(sub_4, [None, None, remainder_3]);  remainder_3 = None
        mul_36 = torch.ops.aten.mul.Tensor(index_10, index_11);  index_10 = index_11 = None
        sub_22 = torch.ops.aten.sub.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
        add_31 = torch.ops.aten.add.Tensor(unsqueeze_23, sub_22);  unsqueeze_23 = sub_22 = None
        mul_37 = torch.ops.aten.mul.Tensor(add_31, div_3)
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_37, [-1], True);  mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_24, 30000);  mul_24 = None
        mul_39 = torch.ops.aten.mul.Tensor(sum_10, 2852.367437761131);  sum_10 = None
        add_32 = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        neg_8 = torch.ops.aten.neg.default(div_3)
        mul_40 = torch.ops.aten.mul.Tensor(add_32, neg_8);  add_32 = neg_8 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, convert_element_type_8);  mul_40 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(convert_element_type_8, [1], True)
        clamp_min_4 = torch.ops.aten.clamp_min.default(sum_11, 1);  sum_11 = None
        div_4 = torch.ops.aten.div.Tensor(mul_41, clamp_min_4);  mul_41 = clamp_min_4 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(div_4, 2)
        sum_12 = torch.ops.aten.sum.dim_IntList(pow_3, [2], True);  pow_3 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sum_12, 0.5);  sum_12 = None
        mul_42 = torch.ops.aten.mul.Tensor(pow_4, 1.0);  pow_4 = None
        full_default_34 = torch.ops.aten.full.default([16, 1023, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False);  full_default_34 = None
        select_22 = torch.ops.aten.select.int(arg6_1, 2, 0)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(select_22, 2)
        sum_13 = torch.ops.aten.sum.dim_IntList(pow_5, [-1], True);  pow_5 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(sum_13, 0.5);  sum_13 = None
        clamp_min_5 = torch.ops.aten.clamp_min.default(pow_6, 1e-06);  pow_6 = None
        div_5 = torch.ops.aten.div.Tensor(select_22, clamp_min_5);  select_22 = clamp_min_5 = None
        clamp_min_6 = torch.ops.aten.clamp_min.default(arg12_1, -1.0)
        clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_6, 1.0);  clamp_min_6 = None
        select_23 = torch.ops.aten.select.int(clamp_max_3, 1, 0)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(select_23, 1);  select_23 = None
        mul_43 = torch.ops.aten.mul.Tensor(unsqueeze_25, div_5);  unsqueeze_25 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(mul_43, 1);  mul_43 = None
        sub_23 = torch.ops.aten.sub.Tensor(unsqueeze_26, add_31);  unsqueeze_26 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_23, div_3)
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_44, [-1], True);  mul_44 = None
        mul_45 = torch.ops.aten.mul.Tensor(sum_14, div_3);  sum_14 = None
        sub_24 = torch.ops.aten.sub.Tensor(sub_23, mul_45);  sub_23 = mul_45 = None
        tanh = torch.ops.aten.tanh.default(sub_24);  sub_24 = None
        select_24 = torch.ops.aten.select.int(arg3_1, 0, 0)
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(select_24, -1);  select_24 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_42, tanh);  tanh = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_27);  mul_46 = unsqueeze_27 = None
        select_25 = torch.ops.aten.select.int(clamp_max_3, 1, 1)
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(select_25, 1);  select_25 = None
        mul_48 = torch.ops.aten.mul.Tensor(unsqueeze_28, div_5);  unsqueeze_28 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(mul_48, 1);  mul_48 = None
        sub_25 = torch.ops.aten.sub.Tensor(unsqueeze_29, add_31);  unsqueeze_29 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_25, div_3)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_49, [-1], True);  mul_49 = None
        mul_50 = torch.ops.aten.mul.Tensor(sum_15, div_3);  sum_15 = None
        sub_26 = torch.ops.aten.sub.Tensor(sub_25, mul_50);  sub_25 = mul_50 = None
        tanh_1 = torch.ops.aten.tanh.default(sub_26);  sub_26 = None
        select_26 = torch.ops.aten.select.int(arg3_1, 0, 1)
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(select_26, -1);  select_26 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_42, tanh_1);  tanh_1 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_30);  mul_51 = unsqueeze_30 = None
        add_34 = torch.ops.aten.add.Tensor(mul_47, mul_52);  mul_47 = mul_52 = None
        select_27 = torch.ops.aten.select.int(clamp_max_3, 1, 2)
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(select_27, 1);  select_27 = None
        mul_53 = torch.ops.aten.mul.Tensor(unsqueeze_31, div_5);  unsqueeze_31 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(mul_53, 1);  mul_53 = None
        sub_27 = torch.ops.aten.sub.Tensor(unsqueeze_32, add_31);  unsqueeze_32 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_27, div_3)
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_54, [-1], True);  mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(sum_16, div_3);  sum_16 = None
        sub_28 = torch.ops.aten.sub.Tensor(sub_27, mul_55);  sub_27 = mul_55 = None
        tanh_2 = torch.ops.aten.tanh.default(sub_28);  sub_28 = None
        select_28 = torch.ops.aten.select.int(arg3_1, 0, 2)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(select_28, -1);  select_28 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_42, tanh_2);  tanh_2 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        add_35 = torch.ops.aten.add.Tensor(add_34, mul_57);  add_34 = mul_57 = None
        select_29 = torch.ops.aten.select.int(clamp_max_3, 1, 3);  clamp_max_3 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(select_29, 1);  select_29 = None
        mul_58 = torch.ops.aten.mul.Tensor(unsqueeze_34, div_5);  unsqueeze_34 = div_5 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(mul_58, 1);  mul_58 = None
        sub_29 = torch.ops.aten.sub.Tensor(unsqueeze_35, add_31);  unsqueeze_35 = add_31 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_29, div_3)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_59, [-1], True);  mul_59 = None
        mul_60 = torch.ops.aten.mul.Tensor(sum_17, div_3);  sum_17 = None
        sub_30 = torch.ops.aten.sub.Tensor(sub_29, mul_60);  sub_29 = mul_60 = None
        tanh_3 = torch.ops.aten.tanh.default(sub_30);  sub_30 = None
        select_30 = torch.ops.aten.select.int(arg3_1, 0, 3);  arg3_1 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(select_30, -1);  select_30 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_42, tanh_3);  mul_42 = tanh_3 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_36);  mul_61 = unsqueeze_36 = None
        add_36 = torch.ops.aten.add.Tensor(add_35, mul_62);  add_35 = mul_62 = None
        add_37 = torch.ops.aten.add.Tensor(div_4, add_36)
        iota_5 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add_38 = torch.ops.aten.add.Tensor(iota_5, 1)
        remainder_4 = torch.ops.aten.remainder.Scalar(add_38, 3);  add_38 = None
        index_12 = torch.ops.aten.index.Tensor(sub_4, [None, None, remainder_4]);  remainder_4 = None
        add_39 = torch.ops.aten.add.Tensor(iota_5, 2)
        remainder_5 = torch.ops.aten.remainder.Scalar(add_39, 3);  add_39 = None
        index_13 = torch.ops.aten.index.Tensor(add_37, [None, None, remainder_5]);  remainder_5 = None
        mul_63 = torch.ops.aten.mul.Tensor(index_12, index_13);  index_12 = index_13 = None
        add_40 = torch.ops.aten.add.Tensor(iota_5, 2)
        remainder_6 = torch.ops.aten.remainder.Scalar(add_40, 3);  add_40 = None
        index_14 = torch.ops.aten.index.Tensor(sub_4, [None, None, remainder_6]);  remainder_6 = None
        add_41 = torch.ops.aten.add.Tensor(iota_5, 1);  iota_5 = None
        remainder_7 = torch.ops.aten.remainder.Scalar(add_41, 3);  add_41 = None
        index_15 = torch.ops.aten.index.Tensor(add_37, [None, None, remainder_7]);  remainder_7 = None
        mul_64 = torch.ops.aten.mul.Tensor(index_14, index_15);  index_14 = index_15 = None
        sub_31 = torch.ops.aten.sub.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(sub_31, [1]);  sub_31 = None
        clamp_min_7 = torch.ops.aten.clamp_min.default(sum_18, -500.0);  sum_18 = None
        clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_7, 500.0);  clamp_min_7 = None
        _linalg_solve_ex = torch.ops.aten._linalg_solve_ex.default(view_10, clamp_max_4)
        getitem = _linalg_solve_ex[0];  _linalg_solve_ex = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(add_37, [1]);  add_37 = None
        add_42 = torch.ops.aten.add.Tensor(lift_fresh_copy, sum_19);  lift_fresh_copy = sum_19 = None
        div_6 = torch.ops.aten.div.Tensor(add_42, 67.8);  add_42 = None
        slice_36 = torch.ops.aten.slice.Tensor(arg12_1, 1, 4, 9223372036854775807);  arg12_1 = None
        neg_9 = torch.ops.aten.neg.default(arg13_1)
        clamp_min_8 = torch.ops.aten.clamp_min.Tensor(slice_36, neg_9);  slice_36 = neg_9 = None
        clamp_max_5 = torch.ops.aten.clamp_max.Tensor(clamp_min_8, arg13_1);  clamp_min_8 = arg13_1 = None
        mul_65 = torch.ops.aten.mul.Tensor(arg10_1, 0.01)
        div_7 = torch.ops.aten.div.Tensor(mul_65, 2)
        add_43 = torch.ops.aten.add.Tensor(arg10_1, div_7);  div_7 = None
        mul_66 = torch.ops.aten.mul.Tensor(add_43, 0.01);  add_43 = None
        div_8 = torch.ops.aten.div.Tensor(mul_66, 2)
        add_44 = torch.ops.aten.add.Tensor(arg10_1, div_8);  div_8 = None
        mul_67 = torch.ops.aten.mul.Tensor(add_44, 0.01);  add_44 = None
        add_45 = torch.ops.aten.add.Tensor(arg10_1, mul_67)
        mul_68 = torch.ops.aten.mul.Tensor(add_45, 0.01);  add_45 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_66, 2);  mul_66 = None
        add_46 = torch.ops.aten.add.Tensor(mul_65, mul_69);  mul_65 = mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_67, 2);  mul_67 = None
        add_47 = torch.ops.aten.add.Tensor(add_46, mul_70);  add_46 = mul_70 = None
        add_48 = torch.ops.aten.add.Tensor(add_47, mul_68);  add_47 = mul_68 = None
        div_9 = torch.ops.aten.div.Tensor(add_48, 6);  add_48 = None
        add_49 = torch.ops.aten.add.Tensor(arg5_1, div_9);  arg5_1 = div_9 = None
        mul_71 = torch.ops.aten.mul.Tensor(div_6, 0.01)
        div_10 = torch.ops.aten.div.Tensor(mul_71, 2)
        add_50 = torch.ops.aten.add.Tensor(div_6, div_10);  div_10 = None
        mul_72 = torch.ops.aten.mul.Tensor(add_50, 0.01);  add_50 = None
        div_11 = torch.ops.aten.div.Tensor(mul_72, 2)
        add_51 = torch.ops.aten.add.Tensor(div_6, div_11);  div_11 = None
        mul_73 = torch.ops.aten.mul.Tensor(add_51, 0.01);  add_51 = None
        add_52 = torch.ops.aten.add.Tensor(div_6, mul_73)
        mul_74 = torch.ops.aten.mul.Tensor(add_52, 0.01);  add_52 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_72, 2);  mul_72 = None
        add_53 = torch.ops.aten.add.Tensor(mul_71, mul_75);  mul_71 = mul_75 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_73, 2);  mul_73 = None
        add_54 = torch.ops.aten.add.Tensor(add_53, mul_76);  add_53 = mul_76 = None
        add_55 = torch.ops.aten.add.Tensor(add_54, mul_74);  add_54 = mul_74 = None
        div_12 = torch.ops.aten.div.Tensor(add_55, 6);  add_55 = None
        add_56 = torch.ops.aten.add.Tensor(arg10_1, div_12);  arg10_1 = div_12 = None
        mul_77 = torch.ops.aten.mul.Tensor(getitem, 0.01)
        div_13 = torch.ops.aten.div.Tensor(mul_77, 2)
        add_57 = torch.ops.aten.add.Tensor(getitem, div_13);  div_13 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_57, 0.01);  add_57 = None
        div_14 = torch.ops.aten.div.Tensor(mul_78, 2)
        add_58 = torch.ops.aten.add.Tensor(getitem, div_14);  div_14 = None
        mul_79 = torch.ops.aten.mul.Tensor(add_58, 0.01);  add_58 = None
        add_59 = torch.ops.aten.add.Tensor(getitem, mul_79)
        mul_80 = torch.ops.aten.mul.Tensor(add_59, 0.01);  add_59 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_78, 2);  mul_78 = None
        add_60 = torch.ops.aten.add.Tensor(mul_77, mul_81);  mul_77 = mul_81 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_79, 2);  mul_79 = None
        add_61 = torch.ops.aten.add.Tensor(add_60, mul_82);  add_60 = mul_82 = None
        add_62 = torch.ops.aten.add.Tensor(add_61, mul_80);  add_61 = mul_80 = None
        div_15 = torch.ops.aten.div.Tensor(add_62, 6);  add_62 = None
        add_63 = torch.ops.aten.add.Tensor(arg11_1, div_15);  arg11_1 = div_15 = None
        mul_83 = torch.ops.aten.mul.Tensor(clamp_max_5, 0.01)
        div_16 = torch.ops.aten.div.Tensor(mul_83, 2)
        add_64 = torch.ops.aten.add.Tensor(clamp_max_5, div_16);  div_16 = None
        mul_84 = torch.ops.aten.mul.Tensor(add_64, 0.01);  add_64 = None
        div_17 = torch.ops.aten.div.Tensor(mul_84, 2)
        add_65 = torch.ops.aten.add.Tensor(clamp_max_5, div_17);  div_17 = None
        mul_85 = torch.ops.aten.mul.Tensor(add_65, 0.01);  add_65 = None
        add_66 = torch.ops.aten.add.Tensor(clamp_max_5, mul_85)
        mul_86 = torch.ops.aten.mul.Tensor(add_66, 0.01);  add_66 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_84, 2);  mul_84 = None
        add_67 = torch.ops.aten.add.Tensor(mul_83, mul_87);  mul_83 = mul_87 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_85, 2);  mul_85 = None
        add_68 = torch.ops.aten.add.Tensor(add_67, mul_88);  add_67 = mul_88 = None
        add_69 = torch.ops.aten.add.Tensor(add_68, mul_86);  add_68 = mul_86 = None
        div_18 = torch.ops.aten.div.Tensor(add_69, 6);  add_69 = None
        add_70 = torch.ops.aten.add.Tensor(arg0_1, div_18);  arg0_1 = div_18 = None
        select_31 = torch.ops.aten.select.int(arg14_1, 0, 0)
        select_32 = torch.ops.aten.select.int(arg14_1, 0, 1);  arg14_1 = None
        clamp_min_9 = torch.ops.aten.clamp_min.Tensor(add_70, select_31);  add_70 = select_31 = None
        clamp_max_6 = torch.ops.aten.clamp_max.Tensor(clamp_min_9, select_32);  clamp_min_9 = select_32 = None
        mul_89 = torch.ops.aten.mul.Tensor(add_63, 0.01)
        div_19 = torch.ops.aten.div.Tensor(mul_89, 2)
        add_71 = torch.ops.aten.add.Tensor(add_63, div_19);  div_19 = None
        mul_90 = torch.ops.aten.mul.Tensor(add_71, 0.01);  add_71 = None
        div_20 = torch.ops.aten.div.Tensor(mul_90, 2)
        add_72 = torch.ops.aten.add.Tensor(add_63, div_20);  div_20 = None
        mul_91 = torch.ops.aten.mul.Tensor(add_72, 0.01);  add_72 = None
        add_73 = torch.ops.aten.add.Tensor(add_63, mul_91)
        mul_92 = torch.ops.aten.mul.Tensor(add_73, 0.01);  add_73 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_90, 2);  mul_90 = None
        add_74 = torch.ops.aten.add.Tensor(mul_89, mul_93);  mul_89 = mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_91, 2);  mul_91 = None
        add_75 = torch.ops.aten.add.Tensor(add_74, mul_94);  add_74 = mul_94 = None
        add_76 = torch.ops.aten.add.Tensor(add_75, mul_92);  add_75 = mul_92 = None
        div_21 = torch.ops.aten.div.Tensor(add_76, 6);  add_76 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(div_21, 2)
        sum_20 = torch.ops.aten.sum.dim_IntList(pow_7, [1], True);  pow_7 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(sum_20, 0.5);  sum_20 = None
        full_default_35 = torch.ops.aten.full.default([16, 3, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_33 = torch.ops.aten.select.int(div_21, 1, 2)
        neg_10 = torch.ops.aten.neg.default(select_33);  select_33 = None
        select_34 = torch.ops.aten.select.int(full_default_35, 1, 0)
        select_35 = torch.ops.aten.select.int(select_34, 1, 1);  select_34 = None
        copy = torch.ops.aten.copy.default(select_35, neg_10);  select_35 = neg_10 = None
        select_36 = torch.ops.aten.select.int(full_default_35, 1, 0)
        select_scatter = torch.ops.aten.select_scatter.default(select_36, copy, 1, 1);  select_36 = copy = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default_35, select_scatter, 1, 0);  full_default_35 = select_scatter = None
        select_39 = torch.ops.aten.select.int(div_21, 1, 1)
        select_42 = torch.ops.aten.select.int(select_scatter_1, 1, 0)
        select_43 = torch.ops.aten.select.int(select_42, 1, 2);  select_42 = None
        copy_1 = torch.ops.aten.copy.default(select_43, select_39);  select_43 = select_39 = None
        select_44 = torch.ops.aten.select.int(select_scatter_1, 1, 0)
        select_scatter_2 = torch.ops.aten.select_scatter.default(select_44, copy_1, 1, 2);  select_44 = copy_1 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(select_scatter_1, select_scatter_2, 1, 0);  select_scatter_1 = select_scatter_2 = None
        select_47 = torch.ops.aten.select.int(div_21, 1, 0)
        neg_11 = torch.ops.aten.neg.default(select_47);  select_47 = None
        select_50 = torch.ops.aten.select.int(select_scatter_3, 1, 1)
        select_51 = torch.ops.aten.select.int(select_50, 1, 2);  select_50 = None
        copy_2 = torch.ops.aten.copy.default(select_51, neg_11);  select_51 = neg_11 = None
        select_52 = torch.ops.aten.select.int(select_scatter_3, 1, 1)
        select_scatter_4 = torch.ops.aten.select_scatter.default(select_52, copy_2, 1, 2);  select_52 = copy_2 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(select_scatter_3, select_scatter_4, 1, 1);  select_scatter_3 = select_scatter_4 = None
        select_55 = torch.ops.aten.select.int(div_21, 1, 2)
        select_58 = torch.ops.aten.select.int(select_scatter_5, 1, 1)
        select_59 = torch.ops.aten.select.int(select_58, 1, 0);  select_58 = None
        copy_3 = torch.ops.aten.copy.default(select_59, select_55);  select_59 = select_55 = None
        select_60 = torch.ops.aten.select.int(select_scatter_5, 1, 1)
        select_scatter_6 = torch.ops.aten.select_scatter.default(select_60, copy_3, 1, 0);  select_60 = copy_3 = None
        select_scatter_7 = torch.ops.aten.select_scatter.default(select_scatter_5, select_scatter_6, 1, 1);  select_scatter_5 = select_scatter_6 = None
        select_63 = torch.ops.aten.select.int(div_21, 1, 1)
        neg_12 = torch.ops.aten.neg.default(select_63);  select_63 = None
        select_66 = torch.ops.aten.select.int(select_scatter_7, 1, 2)
        select_67 = torch.ops.aten.select.int(select_66, 1, 0);  select_66 = None
        copy_4 = torch.ops.aten.copy.default(select_67, neg_12);  select_67 = neg_12 = None
        select_68 = torch.ops.aten.select.int(select_scatter_7, 1, 2)
        select_scatter_8 = torch.ops.aten.select_scatter.default(select_68, copy_4, 1, 0);  select_68 = copy_4 = None
        select_scatter_9 = torch.ops.aten.select_scatter.default(select_scatter_7, select_scatter_8, 1, 2);  select_scatter_7 = select_scatter_8 = None
        select_71 = torch.ops.aten.select.int(div_21, 1, 0);  div_21 = None
        select_74 = torch.ops.aten.select.int(select_scatter_9, 1, 2)
        select_75 = torch.ops.aten.select.int(select_74, 1, 1);  select_74 = None
        copy_5 = torch.ops.aten.copy.default(select_75, select_71);  select_75 = select_71 = None
        select_76 = torch.ops.aten.select.int(select_scatter_9, 1, 2)
        select_scatter_10 = torch.ops.aten.select_scatter.default(select_76, copy_5, 1, 1);  select_76 = copy_5 = None
        select_scatter_11 = torch.ops.aten.select_scatter.default(select_scatter_9, select_scatter_10, 1, 2);  select_scatter_9 = select_scatter_10 = None
        iota_6 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        iota_7 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(iota_6, -1);  iota_6 = None
        eq_1 = torch.ops.aten.eq.Tensor(unsqueeze_37, iota_7);  unsqueeze_37 = iota_7 = None
        full_default_36 = torch.ops.aten.full.default([1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        full_default_37 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_24 = torch.ops.aten.where.self(eq_1, full_default_36, full_default_37);  eq_1 = full_default_36 = full_default_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(where_24, 0);  where_24 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(pow_8, 2);  pow_8 = None
        clamp_min_10 = torch.ops.aten.clamp_min.default(unsqueeze_39, 1e-06);  unsqueeze_39 = None
        sin_4 = torch.ops.aten.sin.default(clamp_min_10)
        div_22 = torch.ops.aten.div.Tensor(sin_4, clamp_min_10);  sin_4 = None
        cos_4 = torch.ops.aten.cos.default(clamp_min_10)
        sub_32 = torch.ops.aten.sub.Tensor(1, cos_4);  cos_4 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(clamp_min_10, 2);  clamp_min_10 = None
        div_23 = torch.ops.aten.div.Tensor(sub_32, pow_9);  sub_32 = pow_9 = None
        bmm_5 = torch.ops.aten.bmm.default(select_scatter_11, select_scatter_11)
        mul_95 = torch.ops.aten.mul.Tensor(div_22, select_scatter_11);  div_22 = select_scatter_11 = None
        add_77 = torch.ops.aten.add.Tensor(unsqueeze_38, mul_95);  unsqueeze_38 = mul_95 = None
        mul_96 = torch.ops.aten.mul.Tensor(div_23, bmm_5);  div_23 = bmm_5 = None
        add_78 = torch.ops.aten.add.Tensor(add_77, mul_96);  add_77 = mul_96 = None
        bmm_6 = torch.ops.aten.bmm.default(add_78, arg6_1);  add_78 = arg6_1 = None
        return (div_6, getitem, clamp_max_5, div_4, add_36, convert_element_type_8, div_3, add_8, add_9, clamp_max_4, div, sub_4, view_10, add_49, add_56, bmm_6, add_63, clamp_max_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 256)
    reader.tensor(buf0, (16, 4), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 196416)
    reader.tensor(buf1, (16, 1023, 3), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 48)
    reader.tensor(buf2, (4, 3), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 16368)
    reader.tensor(buf3, (4, 1023), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 4092)
    reader.tensor(buf4, (1023,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 192)
    reader.tensor(buf5, (16, 3), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 576)
    reader.tensor(buf6, (16, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 4092)
    reader.tensor(buf7, (1023,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4194304)
    reader.tensor(buf8, (16, 256, 256), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 8388608)
    reader.tensor(buf9, (16, 2, 256, 256), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 192)
    reader.tensor(buf10, (16, 3), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 192)
    reader.tensor(buf11, (16, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256000)
    reader.tensor(buf12, (16, 8), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 16)
    reader.tensor(buf13, (4,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 32)
    reader.tensor(buf14, (2, 4), (1, 2), is_leaf=True)  # arg14_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)