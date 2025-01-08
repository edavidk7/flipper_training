# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


cpp_fused_cat_cos_neg_ones_like_sin_stack_sub_zeros_like_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(4LL*x0)];
            auto tmp1 = std::cos(tmp0);
            auto tmp2 = std::sin(tmp0);
            auto tmp3 = decltype(tmp2)(-tmp2);
            out_ptr0[static_cast<int64_t>(3LL*x0)] = tmp1;
            out_ptr1[static_cast<int64_t>(3LL*x0)] = tmp2;
            out_ptr2[static_cast<int64_t>(3LL*x0)] = tmp3;
            out_ptr3[static_cast<int64_t>(3LL*x0)] = tmp1;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr4[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr5[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(1.0);
            out_ptr6[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr7[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr8[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr1[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr9[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr10[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr3[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr11[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16368LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = in_ptr4[static_cast<int64_t>(x1 + (3LL*x0))];
                    auto tmp1 = in_ptr5[static_cast<int64_t>(x1)];
                    auto tmp3 = in_ptr5[static_cast<int64_t>(3LL + x1)];
                    auto tmp5 = in_ptr5[static_cast<int64_t>(6LL + x1)];
                    auto tmp7 = in_ptr5[static_cast<int64_t>(9LL + x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                    auto tmp6 = decltype(tmp0)(tmp0 - tmp5);
                    auto tmp8 = decltype(tmp0)(tmp0 - tmp7);
                    out_ptr12[static_cast<int64_t>(x1 + (3LL*x0))] = tmp2;
                    out_ptr13[static_cast<int64_t>(x1 + (3LL*x0))] = tmp4;
                    out_ptr14[static_cast<int64_t>(x1 + (3LL*x0))] = tmp6;
                    out_ptr15[static_cast<int64_t>(x1 + (3LL*x0))] = tmp8;
                }
            }
        }
    }
}
''')


cpp_fused_cat_cos_neg_ones_like_sin_stack_zeros_like_1 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(1LL + (4LL*x0))];
            auto tmp1 = std::cos(tmp0);
            auto tmp2 = std::sin(tmp0);
            auto tmp3 = decltype(tmp2)(-tmp2);
            out_ptr0[static_cast<int64_t>(3LL*x0)] = tmp1;
            out_ptr1[static_cast<int64_t>(3LL*x0)] = tmp2;
            out_ptr2[static_cast<int64_t>(3LL*x0)] = tmp3;
            out_ptr3[static_cast<int64_t>(3LL*x0)] = tmp1;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr4[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr5[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(1.0);
            out_ptr6[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr7[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr8[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr1[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr9[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr10[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr3[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr11[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
}
''')


cpp_fused_cat_cos_neg_ones_like_sin_stack_zeros_like_2 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL + (4LL*x0))];
            auto tmp1 = std::cos(tmp0);
            auto tmp2 = std::sin(tmp0);
            auto tmp3 = decltype(tmp2)(-tmp2);
            out_ptr0[static_cast<int64_t>(3LL*x0)] = tmp1;
            out_ptr1[static_cast<int64_t>(3LL*x0)] = tmp2;
            out_ptr2[static_cast<int64_t>(3LL*x0)] = tmp3;
            out_ptr3[static_cast<int64_t>(3LL*x0)] = tmp1;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr4[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr5[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(1.0);
            out_ptr6[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr7[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr8[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr1[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr9[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr10[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr3[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr11[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
}
''')


cpp_fused_cat_cos_neg_ones_like_sin_stack_zeros_like_3 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(3LL + (4LL*x0))];
            auto tmp1 = std::cos(tmp0);
            auto tmp2 = std::sin(tmp0);
            auto tmp3 = decltype(tmp2)(-tmp2);
            out_ptr0[static_cast<int64_t>(3LL*x0)] = tmp1;
            out_ptr1[static_cast<int64_t>(3LL*x0)] = tmp2;
            out_ptr2[static_cast<int64_t>(3LL*x0)] = tmp3;
            out_ptr3[static_cast<int64_t>(3LL*x0)] = tmp1;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr4[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr1[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr5[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr6[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(1.0);
            out_ptr7[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr8[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr9[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr10[static_cast<int64_t>(3LL*x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                auto tmp0 = in_ptr3[static_cast<int64_t>(x1 + (3LL*x0))];
                out_ptr11[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
            }
        }
    }
}
''')


cpp_fused_add_mul_4 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(x1)];
                        auto tmp1 = in_out_ptr0[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x2)];
                        auto tmp5 = in_ptr0[static_cast<int64_t>(1023LL + x1)];
                        auto tmp6 = in_ptr2[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp7 = in_ptr1[static_cast<int64_t>(3LL + x2)];
                        auto tmp11 = in_ptr0[static_cast<int64_t>(2046LL + x1)];
                        auto tmp12 = in_ptr3[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp13 = in_ptr1[static_cast<int64_t>(6LL + x2)];
                        auto tmp17 = in_ptr0[static_cast<int64_t>(3069LL + x1)];
                        auto tmp18 = in_ptr4[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp19 = in_ptr1[static_cast<int64_t>(9LL + x2)];
                        auto tmp23 = in_ptr5[static_cast<int64_t>(x1)];
                        auto tmp24 = in_ptr6[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                        auto tmp10 = decltype(tmp4)(tmp4 + tmp9);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = decltype(tmp11)(tmp11 * tmp14);
                        auto tmp16 = decltype(tmp10)(tmp10 + tmp15);
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = decltype(tmp17)(tmp17 * tmp20);
                        auto tmp22 = decltype(tmp16)(tmp16 + tmp21);
                        auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
                        auto tmp26 = decltype(tmp22)(tmp22 + tmp25);
                        in_out_ptr0[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_all_bitwise_and_clamp_div_eq_grid_sampler_2d_le_linalg_cross_linalg_vector_norm_mul_neg_ones_stack_sub_sum_tanh_5 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'int64_t*', 'int64_t*', 'float*', 'int64_t*', 'int64_t*', 'float*', 'int64_t*', 'int64_t*', 'float*', 'float*', 'float*', 'int64_t*', 'int64_t*', 'float*', 'int64_t*', 'int64_t*', 'float*', 'int64_t*', 'int64_t*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17,
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       int64_t* out_ptr22,
                       int64_t* out_ptr23,
                       float* out_ptr24,
                       int64_t* out_ptr25,
                       int64_t* out_ptr26,
                       float* out_ptr27,
                       int64_t* out_ptr28,
                       int64_t* out_ptr29,
                       float* out_ptr30,
                       float* out_ptr32,
                       float* out_ptr33,
                       int64_t* out_ptr34,
                       int64_t* out_ptr35,
                       float* out_ptr36,
                       int64_t* out_ptr37,
                       int64_t* out_ptr38,
                       float* out_ptr39,
                       int64_t* out_ptr40,
                       int64_t* out_ptr41,
                       float* out_ptr42,
                       float* out_ptr43,
                       float* out_ptr45,
                       float* out_ptr46,
                       float* out_ptr47,
                       float* out_ptr48,
                       float* out_ptr49,
                       float* out_ptr50,
                       float* out_ptr51,
                       float* out_ptr52,
                       float* out_ptr53,
                       float* out_ptr54,
                       float* out_ptr55,
                       float* out_ptr56,
                       float* out_ptr57)
{
    auto out_ptr0 = in_out_ptr1;
    auto out_ptr59 = in_out_ptr5;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x2 + (3LL*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        in_out_ptr0[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))] = tmp2;
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(3LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(1023LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp0 = in_ptr1[static_cast<int64_t>(x2)];
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + (3LL*x2) + (3069LL*x0)), static_cast<int64_t>(3LL));
                            auto tmp2 = at::vec::Vectorized<float>(tmp0);
                            auto tmp3 = tmp2 * tmp1;
                            tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp3, static_cast<int64_t>(3LL));
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x1 + (3LL*x0)), static_cast<int64_t>(3LL));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1016LL); x0+=static_cast<int64_t>(8LL))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    for(int64_t x0=static_cast<int64_t>(1016LL); x0<static_cast<int64_t>(1023LL); x0+=static_cast<int64_t>(7LL))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(7LL));
                        tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp0, static_cast<int64_t>(7LL));
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<int64_t>(0LL)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(48LL); x0+=static_cast<int64_t>(8LL))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = out_ptr1[static_cast<int64_t>(0LL)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<int64_t>(x0));
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp1 = in_out_ptr1[static_cast<int64_t>(x2 + (3LL*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        out_ptr2[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))] = tmp2;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(8LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc2 = 0;
                        at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc3 = 0;
                        at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc4 = 0;
                        at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc5 = 0;
                        at::vec::Vectorized<float> tmp_acc5_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1016LL); x1+=static_cast<int64_t>(8LL))
                        {
                            for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                            {
                                auto tmp0 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr2[static_cast<int64_t>(1LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + x1_inner)];
                                auto tmp5 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr2[static_cast<int64_t>(2LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp11 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr2[static_cast<int64_t>((3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp1 = tmp0 * tmp0;
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp6 = tmp5 * tmp5;
                                auto tmp7 = tmp6 * tmp3;
                                auto tmp8 = tmp4 + tmp7;
                                auto tmp9 = tmp3 * tmp0;
                                auto tmp10 = tmp9 * tmp5;
                                auto tmp12 = tmp3 * tmp11;
                                auto tmp13 = tmp12 * tmp0;
                                auto tmp14 = tmp11 * tmp11;
                                auto tmp15 = tmp14 * tmp3;
                                auto tmp16 = tmp15 + tmp4;
                                auto tmp17 = tmp12 * tmp5;
                                auto tmp18 = tmp15 + tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                                tmp_acc1_vec = tmp_acc1_vec + tmp10;
                                tmp_acc2_vec = tmp_acc2_vec + tmp13;
                                tmp_acc3_vec = tmp_acc3_vec + tmp16;
                                tmp_acc4_vec = tmp_acc4_vec + tmp17;
                                tmp_acc5_vec = tmp_acc5_vec + tmp18;
                            }
                        }
                        for(int64_t x1=static_cast<int64_t>(1016LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(7LL))
                        {
                            for (long x1_inner = 0; x1_inner < static_cast<int64_t>(7LL); x1_inner++)
                            {
                                auto tmp0 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr2[static_cast<int64_t>(1LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + x1_inner)];
                                auto tmp5 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr2[static_cast<int64_t>(2LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp11 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr2[static_cast<int64_t>((3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp1 = tmp0 * tmp0;
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp6 = tmp5 * tmp5;
                                auto tmp7 = tmp6 * tmp3;
                                auto tmp8 = tmp4 + tmp7;
                                auto tmp9 = tmp3 * tmp0;
                                auto tmp10 = tmp9 * tmp5;
                                auto tmp12 = tmp3 * tmp11;
                                auto tmp13 = tmp12 * tmp0;
                                auto tmp14 = tmp11 * tmp11;
                                auto tmp15 = tmp14 * tmp3;
                                auto tmp16 = tmp15 + tmp4;
                                auto tmp17 = tmp12 * tmp5;
                                auto tmp18 = tmp15 + tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                                tmp_acc1_vec = tmp_acc1_vec + tmp10;
                                tmp_acc2_vec = tmp_acc2_vec + tmp13;
                                tmp_acc3_vec = tmp_acc3_vec + tmp16;
                                tmp_acc4_vec = tmp_acc4_vec + tmp17;
                                tmp_acc5_vec = tmp_acc5_vec + tmp18;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr4 + static_cast<int64_t>(x0));
                        tmp_acc2_vec.store(out_ptr5 + static_cast<int64_t>(x0));
                        tmp_acc3_vec.store(out_ptr6 + static_cast<int64_t>(x0));
                        tmp_acc4_vec.store(out_ptr7 + static_cast<int64_t>(x0));
                        tmp_acc5_vec.store(out_ptr8 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = out_ptr5[static_cast<int64_t>(x0)];
                    auto tmp1 = decltype(tmp0)(-tmp0);
                    out_ptr9[static_cast<int64_t>(3LL*x0)] = tmp1;
                    out_ptr10[static_cast<int64_t>(3LL*x0)] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = out_ptr3[static_cast<int64_t>(x0)];
                    out_ptr11[static_cast<int64_t>(3LL*x0)] = tmp0;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = out_ptr8[static_cast<int64_t>(x0)];
                    out_ptr12[static_cast<int64_t>(3LL*x0)] = tmp0;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = out_ptr7[static_cast<int64_t>(x0)];
                    auto tmp1 = decltype(tmp0)(-tmp0);
                    out_ptr13[static_cast<int64_t>(3LL*x0)] = tmp1;
                    out_ptr14[static_cast<int64_t>(3LL*x0)] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = out_ptr4[static_cast<int64_t>(x0)];
                    auto tmp1 = decltype(tmp0)(-tmp0);
                    out_ptr15[static_cast<int64_t>(3LL*x0)] = tmp1;
                    out_ptr16[static_cast<int64_t>(3LL*x0)] = tmp1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = out_ptr6[static_cast<int64_t>(x0)];
                    out_ptr17[static_cast<int64_t>(3LL*x0)] = tmp0;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + (3LL*x0))];
                        out_ptr18[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_ptr3[static_cast<int64_t>(x1 + (3LL*x0))];
                        out_ptr19[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_ptr4[static_cast<int64_t>(x1 + (3LL*x0))];
                        out_ptr20[static_cast<int64_t>(x1 + (9LL*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16368LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(3LL*x0)];
                    auto tmp15 = in_out_ptr0[static_cast<int64_t>(1LL + (3LL*x0))];
                    auto tmp1 = static_cast<float>(0.15625);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp3 = static_cast<float>(-1.0);
                    auto tmp4 = max_propagate_nan(tmp2, tmp3);
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = min_propagate_nan(tmp4, tmp5);
                    auto tmp7 = static_cast<float>(127.5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = decltype(tmp8)(tmp8 + tmp7);
                    auto tmp10 = std::floor(tmp9);
                    auto tmp11 = static_cast<float>(0.0);
                    auto tmp12 = tmp10 >= tmp11;
                    auto tmp13 = static_cast<float>(256.0);
                    auto tmp14 = tmp10 < tmp13;
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp1);
                    auto tmp17 = max_propagate_nan(tmp16, tmp3);
                    auto tmp18 = min_propagate_nan(tmp17, tmp5);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp7);
                    auto tmp20 = decltype(tmp19)(tmp19 + tmp7);
                    auto tmp21 = std::floor(tmp20);
                    auto tmp22 = tmp21 >= tmp11;
                    auto tmp23 = tmp21 < tmp13;
                    auto tmp24 = tmp22 && tmp23;
                    auto tmp25 = tmp14 && tmp24;
                    auto tmp26 = tmp12 && tmp25;
                    auto tmp27 = decltype(tmp10)(tmp10 + tmp5);
                    auto tmp28 = decltype(tmp27)(tmp27 - tmp9);
                    auto tmp29 = decltype(tmp21)(tmp21 + tmp5);
                    auto tmp30 = decltype(tmp29)(tmp29 - tmp20);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = tmp26 ? tmp31 : tmp11;
                    auto tmp33 = tmp27 >= tmp11;
                    auto tmp34 = tmp27 < tmp13;
                    auto tmp35 = tmp34 && tmp24;
                    auto tmp36 = tmp33 && tmp35;
                    auto tmp37 = c10::convert<int64_t>(tmp21);
                    auto tmp38 = static_cast<int64_t>(0);
                    auto tmp39 = tmp36 ? tmp37 : tmp38;
                    auto tmp40 = c10::convert<int64_t>(tmp27);
                    auto tmp41 = tmp36 ? tmp40 : tmp38;
                    auto tmp42 = decltype(tmp9)(tmp9 - tmp10);
                    auto tmp43 = decltype(tmp42)(tmp42 * tmp30);
                    auto tmp44 = tmp36 ? tmp43 : tmp11;
                    auto tmp45 = tmp29 >= tmp11;
                    auto tmp46 = tmp29 < tmp13;
                    auto tmp47 = tmp45 && tmp46;
                    auto tmp48 = tmp14 && tmp47;
                    auto tmp49 = tmp12 && tmp48;
                    auto tmp50 = c10::convert<int64_t>(tmp29);
                    auto tmp51 = tmp49 ? tmp50 : tmp38;
                    auto tmp52 = c10::convert<int64_t>(tmp10);
                    auto tmp53 = tmp49 ? tmp52 : tmp38;
                    auto tmp54 = decltype(tmp20)(tmp20 - tmp21);
                    auto tmp55 = decltype(tmp28)(tmp28 * tmp54);
                    auto tmp56 = tmp49 ? tmp55 : tmp11;
                    auto tmp57 = tmp34 && tmp47;
                    auto tmp58 = tmp33 && tmp57;
                    auto tmp59 = tmp58 ? tmp50 : tmp38;
                    auto tmp60 = tmp58 ? tmp40 : tmp38;
                    auto tmp61 = decltype(tmp42)(tmp42 * tmp54);
                    auto tmp62 = tmp58 ? tmp61 : tmp11;
                    out_ptr21[static_cast<int64_t>(x0)] = tmp32;
                    out_ptr22[static_cast<int64_t>(x0)] = tmp39;
                    out_ptr23[static_cast<int64_t>(x0)] = tmp41;
                    out_ptr24[static_cast<int64_t>(x0)] = tmp44;
                    out_ptr25[static_cast<int64_t>(x0)] = tmp51;
                    out_ptr26[static_cast<int64_t>(x0)] = tmp53;
                    out_ptr27[static_cast<int64_t>(x0)] = tmp56;
                    out_ptr28[static_cast<int64_t>(x0)] = tmp59;
                    out_ptr29[static_cast<int64_t>(x0)] = tmp60;
                    out_ptr30[static_cast<int64_t>(x0)] = tmp62;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<int64_t>((3LL*x1) + (3069LL*x0))];
                        auto tmp15 = in_out_ptr0[static_cast<int64_t>(1LL + (3LL*x1) + (3069LL*x0))];
                        auto tmp47 = out_ptr21[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp49 = out_ptr22[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp56 = out_ptr23[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp64 = out_ptr24[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp67 = out_ptr25[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp74 = out_ptr26[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp82 = out_ptr27[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp85 = in_out_ptr0[static_cast<int64_t>(2LL + (3LL*x1) + (3069LL*x0))];
                        auto tmp86 = out_ptr28[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp93 = out_ptr29[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp101 = out_ptr30[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp1 = static_cast<float>(0.15625);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(-1.0);
                        auto tmp4 = max_propagate_nan(tmp2, tmp3);
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = min_propagate_nan(tmp4, tmp5);
                        auto tmp7 = static_cast<float>(127.5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp9 = decltype(tmp8)(tmp8 + tmp7);
                        auto tmp10 = std::floor(tmp9);
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = static_cast<float>(256.0);
                        auto tmp14 = tmp10 < tmp13;
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp1);
                        auto tmp17 = max_propagate_nan(tmp16, tmp3);
                        auto tmp18 = min_propagate_nan(tmp17, tmp5);
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp7);
                        auto tmp20 = decltype(tmp19)(tmp19 + tmp7);
                        auto tmp21 = std::floor(tmp20);
                        auto tmp22 = tmp21 >= tmp11;
                        auto tmp23 = tmp21 < tmp13;
                        auto tmp24 = tmp22 && tmp23;
                        auto tmp25 = tmp14 && tmp24;
                        auto tmp26 = tmp12 && tmp25;
                        auto tmp27 = c10::convert<int64_t>(tmp21);
                        auto tmp28 = static_cast<int64_t>(0);
                        auto tmp29 = tmp26 ? tmp27 : tmp28;
                        auto tmp30 = 256LL;
                        auto tmp31 = c10::convert<int64_t>(tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 + tmp31);
                        auto tmp33 = tmp29 < 0;
                        auto tmp34 = tmp33 ? tmp32 : tmp29;
                        auto tmp35 = tmp34;
                        auto tmp36 = c10::convert<int64_t>(tmp35);
                        TORCH_CHECK((0 <= tmp36) & (tmp36 < 256LL), "index out of bounds: 0 <= tmp36 < 256LL");
                        auto tmp38 = c10::convert<int64_t>(tmp10);
                        auto tmp39 = tmp26 ? tmp38 : tmp28;
                        auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                        auto tmp41 = tmp39 < 0;
                        auto tmp42 = tmp41 ? tmp40 : tmp39;
                        auto tmp43 = tmp42;
                        auto tmp44 = c10::convert<int64_t>(tmp43);
                        TORCH_CHECK((0 <= tmp44) & (tmp44 < 256LL), "index out of bounds: 0 <= tmp44 < 256LL");
                        auto tmp46 = in_ptr5[static_cast<int64_t>(tmp42 + (256LL*tmp34) + (65536LL*x0))];
                        auto tmp48 = decltype(tmp46)(tmp46 * tmp47);
                        auto tmp50 = decltype(tmp49)(tmp49 + tmp31);
                        auto tmp51 = tmp49 < 0;
                        auto tmp52 = tmp51 ? tmp50 : tmp49;
                        auto tmp53 = tmp52;
                        auto tmp54 = c10::convert<int64_t>(tmp53);
                        TORCH_CHECK((0 <= tmp54) & (tmp54 < 256LL), "index out of bounds: 0 <= tmp54 < 256LL");
                        auto tmp57 = decltype(tmp56)(tmp56 + tmp31);
                        auto tmp58 = tmp56 < 0;
                        auto tmp59 = tmp58 ? tmp57 : tmp56;
                        auto tmp60 = tmp59;
                        auto tmp61 = c10::convert<int64_t>(tmp60);
                        TORCH_CHECK((0 <= tmp61) & (tmp61 < 256LL), "index out of bounds: 0 <= tmp61 < 256LL");
                        auto tmp63 = in_ptr5[static_cast<int64_t>(tmp59 + (256LL*tmp52) + (65536LL*x0))];
                        auto tmp65 = decltype(tmp63)(tmp63 * tmp64);
                        auto tmp66 = decltype(tmp48)(tmp48 + tmp65);
                        auto tmp68 = decltype(tmp67)(tmp67 + tmp31);
                        auto tmp69 = tmp67 < 0;
                        auto tmp70 = tmp69 ? tmp68 : tmp67;
                        auto tmp71 = tmp70;
                        auto tmp72 = c10::convert<int64_t>(tmp71);
                        TORCH_CHECK((0 <= tmp72) & (tmp72 < 256LL), "index out of bounds: 0 <= tmp72 < 256LL");
                        auto tmp75 = decltype(tmp74)(tmp74 + tmp31);
                        auto tmp76 = tmp74 < 0;
                        auto tmp77 = tmp76 ? tmp75 : tmp74;
                        auto tmp78 = tmp77;
                        auto tmp79 = c10::convert<int64_t>(tmp78);
                        TORCH_CHECK((0 <= tmp79) & (tmp79 < 256LL), "index out of bounds: 0 <= tmp79 < 256LL");
                        auto tmp81 = in_ptr5[static_cast<int64_t>(tmp77 + (256LL*tmp70) + (65536LL*x0))];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = decltype(tmp66)(tmp66 + tmp83);
                        auto tmp87 = decltype(tmp86)(tmp86 + tmp31);
                        auto tmp88 = tmp86 < 0;
                        auto tmp89 = tmp88 ? tmp87 : tmp86;
                        auto tmp90 = tmp89;
                        auto tmp91 = c10::convert<int64_t>(tmp90);
                        TORCH_CHECK((0 <= tmp91) & (tmp91 < 256LL), "index out of bounds: 0 <= tmp91 < 256LL");
                        auto tmp94 = decltype(tmp93)(tmp93 + tmp31);
                        auto tmp95 = tmp93 < 0;
                        auto tmp96 = tmp95 ? tmp94 : tmp93;
                        auto tmp97 = tmp96;
                        auto tmp98 = c10::convert<int64_t>(tmp97);
                        TORCH_CHECK((0 <= tmp98) & (tmp98 < 256LL), "index out of bounds: 0 <= tmp98 < 256LL");
                        auto tmp100 = in_ptr5[static_cast<int64_t>(tmp96 + (256LL*tmp89) + (65536LL*x0))];
                        auto tmp102 = decltype(tmp100)(tmp100 * tmp101);
                        auto tmp103 = decltype(tmp84)(tmp84 + tmp102);
                        auto tmp104 = decltype(tmp85)(tmp85 - tmp103);
                        in_out_ptr2[static_cast<int64_t>(x1 + (1023LL*x0))] = tmp104;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16368LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = in_out_ptr2[static_cast<int64_t>(x0)];
                    auto tmp3 = in_out_ptr0[static_cast<int64_t>(3LL*x0)];
                    auto tmp12 = in_out_ptr0[static_cast<int64_t>(1LL + (3LL*x0))];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = static_cast<float>(-6.4);
                    auto tmp5 = max_propagate_nan(tmp3, tmp4);
                    auto tmp6 = static_cast<float>(6.4);
                    auto tmp7 = min_propagate_nan(tmp5, tmp6);
                    auto tmp8 = tmp7 == tmp3;
                    auto tmp9 = !tmp8;
                    auto tmp10 = c10::convert<int64_t>(tmp9);
                    auto tmp11 = c10::convert<bool>(tmp10);
                    auto tmp13 = max_propagate_nan(tmp12, tmp4);
                    auto tmp14 = min_propagate_nan(tmp13, tmp6);
                    auto tmp15 = tmp14 == tmp12;
                    auto tmp16 = !tmp15;
                    auto tmp17 = c10::convert<int64_t>(tmp16);
                    auto tmp18 = c10::convert<bool>(tmp17);
                    auto tmp19 = tmp11 || tmp18;
                    auto tmp20 = !tmp19;
                    auto tmp21 = decltype(tmp2)(tmp2 & tmp20);
                    auto tmp22 = c10::convert<float>(tmp21);
                    auto tmp23 = static_cast<float>(0.15625);
                    auto tmp24 = decltype(tmp3)(tmp3 * tmp23);
                    auto tmp25 = static_cast<float>(-1.0);
                    auto tmp26 = max_propagate_nan(tmp24, tmp25);
                    auto tmp27 = static_cast<float>(1.0);
                    auto tmp28 = min_propagate_nan(tmp26, tmp27);
                    auto tmp29 = static_cast<float>(127.5);
                    auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                    auto tmp31 = decltype(tmp30)(tmp30 + tmp29);
                    auto tmp32 = std::floor(tmp31);
                    auto tmp33 = tmp32 >= tmp1;
                    auto tmp34 = static_cast<float>(256.0);
                    auto tmp35 = tmp32 < tmp34;
                    auto tmp36 = decltype(tmp12)(tmp12 * tmp23);
                    auto tmp37 = max_propagate_nan(tmp36, tmp25);
                    auto tmp38 = min_propagate_nan(tmp37, tmp27);
                    auto tmp39 = decltype(tmp38)(tmp38 * tmp29);
                    auto tmp40 = decltype(tmp39)(tmp39 + tmp29);
                    auto tmp41 = std::floor(tmp40);
                    auto tmp42 = tmp41 >= tmp1;
                    auto tmp43 = tmp41 < tmp34;
                    auto tmp44 = tmp42 && tmp43;
                    auto tmp45 = tmp35 && tmp44;
                    auto tmp46 = tmp33 && tmp45;
                    auto tmp47 = decltype(tmp32)(tmp32 + tmp27);
                    auto tmp48 = decltype(tmp47)(tmp47 - tmp31);
                    auto tmp49 = decltype(tmp41)(tmp41 + tmp27);
                    auto tmp50 = decltype(tmp49)(tmp49 - tmp40);
                    auto tmp51 = decltype(tmp48)(tmp48 * tmp50);
                    auto tmp52 = tmp46 ? tmp51 : tmp1;
                    auto tmp53 = tmp47 >= tmp1;
                    auto tmp54 = tmp47 < tmp34;
                    auto tmp55 = tmp54 && tmp44;
                    auto tmp56 = tmp53 && tmp55;
                    auto tmp57 = c10::convert<int64_t>(tmp41);
                    auto tmp58 = static_cast<int64_t>(0);
                    auto tmp59 = tmp56 ? tmp57 : tmp58;
                    auto tmp60 = c10::convert<int64_t>(tmp47);
                    auto tmp61 = tmp56 ? tmp60 : tmp58;
                    auto tmp62 = decltype(tmp31)(tmp31 - tmp32);
                    auto tmp63 = decltype(tmp62)(tmp62 * tmp50);
                    auto tmp64 = tmp56 ? tmp63 : tmp1;
                    auto tmp65 = tmp49 >= tmp1;
                    auto tmp66 = tmp49 < tmp34;
                    auto tmp67 = tmp65 && tmp66;
                    auto tmp68 = tmp35 && tmp67;
                    auto tmp69 = tmp33 && tmp68;
                    auto tmp70 = c10::convert<int64_t>(tmp49);
                    auto tmp71 = tmp69 ? tmp70 : tmp58;
                    auto tmp72 = c10::convert<int64_t>(tmp32);
                    auto tmp73 = tmp69 ? tmp72 : tmp58;
                    auto tmp74 = decltype(tmp40)(tmp40 - tmp41);
                    auto tmp75 = decltype(tmp48)(tmp48 * tmp74);
                    auto tmp76 = tmp69 ? tmp75 : tmp1;
                    auto tmp77 = tmp54 && tmp67;
                    auto tmp78 = tmp53 && tmp77;
                    auto tmp79 = tmp78 ? tmp70 : tmp58;
                    auto tmp80 = tmp78 ? tmp60 : tmp58;
                    auto tmp81 = decltype(tmp62)(tmp62 * tmp74);
                    auto tmp82 = tmp78 ? tmp81 : tmp1;
                    out_ptr32[static_cast<int64_t>(x0)] = tmp22;
                    out_ptr33[static_cast<int64_t>(x0)] = tmp52;
                    out_ptr34[static_cast<int64_t>(x0)] = tmp59;
                    out_ptr35[static_cast<int64_t>(x0)] = tmp61;
                    out_ptr36[static_cast<int64_t>(x0)] = tmp64;
                    out_ptr37[static_cast<int64_t>(x0)] = tmp71;
                    out_ptr38[static_cast<int64_t>(x0)] = tmp73;
                    out_ptr39[static_cast<int64_t>(x0)] = tmp76;
                    out_ptr40[static_cast<int64_t>(x0)] = tmp79;
                    out_ptr41[static_cast<int64_t>(x0)] = tmp80;
                    out_ptr42[static_cast<int64_t>(x0)] = tmp82;
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_ptr6[static_cast<int64_t>(x2 + (3LL*x0))];
                        auto tmp1 = in_ptr7[static_cast<int64_t>((3LL*x0) + (static_cast<int64_t>((1LL + x2)) % static_cast<int64_t>(3LL)))];
                        auto tmp2 = out_ptr2[static_cast<int64_t>((3LL*x1) + (3069LL*x0) + (static_cast<int64_t>((2LL + x2)) % static_cast<int64_t>(3LL)))];
                        auto tmp4 = in_ptr7[static_cast<int64_t>((3LL*x0) + (static_cast<int64_t>((2LL + x2)) % static_cast<int64_t>(3LL)))];
                        auto tmp5 = out_ptr2[static_cast<int64_t>((3LL*x1) + (3069LL*x0) + (static_cast<int64_t>((1LL + x2)) % static_cast<int64_t>(3LL)))];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp3)(tmp3 - tmp6);
                        auto tmp8 = decltype(tmp0)(tmp0 + tmp7);
                        out_ptr43[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))] = tmp8;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(2LL); x1+=static_cast<int64_t>(1LL))
                    {
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(1023LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<int64_t>((3LL*x2) + (3069LL*x0))];
                            auto tmp15 = in_out_ptr0[static_cast<int64_t>(1LL + (3LL*x2) + (3069LL*x0))];
                            auto tmp47 = out_ptr33[static_cast<int64_t>(x2 + (1023LL*x0))];
                            auto tmp49 = out_ptr34[static_cast<int64_t>(x2 + (1023LL*x0))];
                            auto tmp56 = out_ptr35[static_cast<int64_t>(x2 + (1023LL*x0))];
                            auto tmp64 = out_ptr36[static_cast<int64_t>(x2 + (1023LL*x0))];
                            auto tmp67 = out_ptr37[static_cast<int64_t>(x2 + (1023LL*x0))];
                            auto tmp74 = out_ptr38[static_cast<int64_t>(x2 + (1023LL*x0))];
                            auto tmp82 = out_ptr39[static_cast<int64_t>(x2 + (1023LL*x0))];
                            auto tmp1 = static_cast<float>(0.15625);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = static_cast<float>(-1.0);
                            auto tmp4 = max_propagate_nan(tmp2, tmp3);
                            auto tmp5 = static_cast<float>(1.0);
                            auto tmp6 = min_propagate_nan(tmp4, tmp5);
                            auto tmp7 = static_cast<float>(127.5);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = decltype(tmp8)(tmp8 + tmp7);
                            auto tmp10 = std::floor(tmp9);
                            auto tmp11 = static_cast<float>(0.0);
                            auto tmp12 = tmp10 >= tmp11;
                            auto tmp13 = static_cast<float>(256.0);
                            auto tmp14 = tmp10 < tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp1);
                            auto tmp17 = max_propagate_nan(tmp16, tmp3);
                            auto tmp18 = min_propagate_nan(tmp17, tmp5);
                            auto tmp19 = decltype(tmp18)(tmp18 * tmp7);
                            auto tmp20 = decltype(tmp19)(tmp19 + tmp7);
                            auto tmp21 = std::floor(tmp20);
                            auto tmp22 = tmp21 >= tmp11;
                            auto tmp23 = tmp21 < tmp13;
                            auto tmp24 = tmp22 && tmp23;
                            auto tmp25 = tmp14 && tmp24;
                            auto tmp26 = tmp12 && tmp25;
                            auto tmp27 = c10::convert<int64_t>(tmp21);
                            auto tmp28 = static_cast<int64_t>(0);
                            auto tmp29 = tmp26 ? tmp27 : tmp28;
                            auto tmp30 = 256LL;
                            auto tmp31 = c10::convert<int64_t>(tmp30);
                            auto tmp32 = decltype(tmp29)(tmp29 + tmp31);
                            auto tmp33 = tmp29 < 0;
                            auto tmp34 = tmp33 ? tmp32 : tmp29;
                            auto tmp35 = tmp34;
                            auto tmp36 = c10::convert<int64_t>(tmp35);
                            TORCH_CHECK((0 <= tmp36) & (tmp36 < 256LL), "index out of bounds: 0 <= tmp36 < 256LL");
                            auto tmp38 = c10::convert<int64_t>(tmp10);
                            auto tmp39 = tmp26 ? tmp38 : tmp28;
                            auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                            auto tmp41 = tmp39 < 0;
                            auto tmp42 = tmp41 ? tmp40 : tmp39;
                            auto tmp43 = tmp42;
                            auto tmp44 = c10::convert<int64_t>(tmp43);
                            TORCH_CHECK((0 <= tmp44) & (tmp44 < 256LL), "index out of bounds: 0 <= tmp44 < 256LL");
                            auto tmp46 = in_ptr8[static_cast<int64_t>(tmp42 + (256LL*tmp34) + (65536LL*x1) + (131072LL*x0))];
                            auto tmp48 = decltype(tmp46)(tmp46 * tmp47);
                            auto tmp50 = decltype(tmp49)(tmp49 + tmp31);
                            auto tmp51 = tmp49 < 0;
                            auto tmp52 = tmp51 ? tmp50 : tmp49;
                            auto tmp53 = tmp52;
                            auto tmp54 = c10::convert<int64_t>(tmp53);
                            TORCH_CHECK((0 <= tmp54) & (tmp54 < 256LL), "index out of bounds: 0 <= tmp54 < 256LL");
                            auto tmp57 = decltype(tmp56)(tmp56 + tmp31);
                            auto tmp58 = tmp56 < 0;
                            auto tmp59 = tmp58 ? tmp57 : tmp56;
                            auto tmp60 = tmp59;
                            auto tmp61 = c10::convert<int64_t>(tmp60);
                            TORCH_CHECK((0 <= tmp61) & (tmp61 < 256LL), "index out of bounds: 0 <= tmp61 < 256LL");
                            auto tmp63 = in_ptr8[static_cast<int64_t>(tmp59 + (256LL*tmp52) + (65536LL*x1) + (131072LL*x0))];
                            auto tmp65 = decltype(tmp63)(tmp63 * tmp64);
                            auto tmp66 = decltype(tmp48)(tmp48 + tmp65);
                            auto tmp68 = decltype(tmp67)(tmp67 + tmp31);
                            auto tmp69 = tmp67 < 0;
                            auto tmp70 = tmp69 ? tmp68 : tmp67;
                            auto tmp71 = tmp70;
                            auto tmp72 = c10::convert<int64_t>(tmp71);
                            TORCH_CHECK((0 <= tmp72) & (tmp72 < 256LL), "index out of bounds: 0 <= tmp72 < 256LL");
                            auto tmp75 = decltype(tmp74)(tmp74 + tmp31);
                            auto tmp76 = tmp74 < 0;
                            auto tmp77 = tmp76 ? tmp75 : tmp74;
                            auto tmp78 = tmp77;
                            auto tmp79 = c10::convert<int64_t>(tmp78);
                            TORCH_CHECK((0 <= tmp79) & (tmp79 < 256LL), "index out of bounds: 0 <= tmp79 < 256LL");
                            auto tmp81 = in_ptr8[static_cast<int64_t>(tmp77 + (256LL*tmp70) + (65536LL*x1) + (131072LL*x0))];
                            auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                            auto tmp84 = decltype(tmp66)(tmp66 + tmp83);
                            in_out_ptr3[static_cast<int64_t>(x2 + (1023LL*x1) + (2046LL*x0))] = tmp84;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1016LL); x1+=static_cast<int64_t>(8LL))
                    {
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(2LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<int64_t>(x1 + (1023LL*x2) + (2046LL*x0)), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::VectorizedN<int64_t,2>::loadu(out_ptr40 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(8));
                            auto tmp13 = at::vec::VectorizedN<int64_t,2>::loadu(out_ptr41 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(8));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr42 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(8));
                            auto tmp2 = 256LL;
                            auto tmp3 = c10::convert<int64_t>(tmp2);
                            auto tmp4 = at::vec::VectorizedN<int64_t,2>(tmp3);
                            auto tmp5 = tmp1 + tmp4;
                            auto tmp6 = static_cast<int64_t>(0);
                            auto tmp7 = at::vec::VectorizedN<int64_t,2>(tmp6);
                            auto tmp8 = at::vec::VecMask<int64_t,2>(tmp1 < tmp7);
                            auto tmp9 = decltype(tmp5)::blendv(tmp1, tmp5, tmp8.template cast<int64_t,2>());
                            auto tmp10 =
                            [&]
                            {
                                __at_align__ std::array<int64_t, 8> tmpbuf;
                                tmp9.store(tmpbuf.data(), static_cast<int64_t>(8));
                                return tmpbuf;
                            }
                            ()
                            ;
                            auto tmp11 =
                            [&]
                            {
                                __at_align__ std::array<int64_t, 8> tmpbuf;
                                #pragma unroll 8
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    tmpbuf[x1_inner] = static_cast<int64_t>(tmp10[x1_inner]);
                                }
                                return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                            }
                            ()
                            ;
                            TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp11) & (tmp11 < at::vec::VectorizedN<int64_t,2>(256LL)))).all_masked(), "index out of bounds: 0 <= tmp11 < 256LL");
                            auto tmp14 = tmp13 + tmp4;
                            auto tmp15 = at::vec::VecMask<int64_t,2>(tmp13 < tmp7);
                            auto tmp16 = decltype(tmp14)::blendv(tmp13, tmp14, tmp15.template cast<int64_t,2>());
                            auto tmp17 =
                            [&]
                            {
                                __at_align__ std::array<int64_t, 8> tmpbuf;
                                tmp16.store(tmpbuf.data(), static_cast<int64_t>(8));
                                return tmpbuf;
                            }
                            ()
                            ;
                            auto tmp18 =
                            [&]
                            {
                                __at_align__ std::array<int64_t, 8> tmpbuf;
                                #pragma unroll 8
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    tmpbuf[x1_inner] = static_cast<int64_t>(tmp17[x1_inner]);
                                }
                                return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                            }
                            ()
                            ;
                            TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp18) & (tmp18 < at::vec::VectorizedN<int64_t,2>(256LL)))).all_masked(), "index out of bounds: 0 <= tmp18 < 256LL");
                            auto tmp20 =
                            [&]
                            {
                                __at_align__ std::array<float, 8> tmpbuf;
                                #pragma unroll 8
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    tmpbuf[x1_inner] = in_ptr8[static_cast<int64_t>(tmp17[x1_inner] + (256LL*tmp10[x1_inner]) + (65536LL*x2) + (131072LL*x0))];
                                }
                                return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                            }
                            ()
                            ;
                            auto tmp22 = tmp20 * tmp21;
                            auto tmp23 = tmp0 + tmp22;
                            auto tmp24 = tmp23.neg();
                            [&]
                            {
                                __at_align__ std::array<float, 8> tmpbuf;
                                tmp24.store(tmpbuf.data(), static_cast<int64_t>(8));
                                #pragma unroll 8
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    out_ptr45[static_cast<int64_t>(x2 + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0))] = tmpbuf[x1_inner];
                                }
                            }
                            ()
                            ;
                        }
                    }
                    for(int64_t x1=static_cast<int64_t>(1016LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                    {
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(2LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp0 = in_out_ptr3[static_cast<int64_t>(x1 + (1023LL*x2) + (2046LL*x0))];
                            auto tmp1 = out_ptr40[static_cast<int64_t>(x1 + (1023LL*x0))];
                            auto tmp10 = out_ptr41[static_cast<int64_t>(x1 + (1023LL*x0))];
                            auto tmp18 = out_ptr42[static_cast<int64_t>(x1 + (1023LL*x0))];
                            auto tmp2 = 256LL;
                            auto tmp3 = c10::convert<int64_t>(tmp2);
                            auto tmp4 = decltype(tmp1)(tmp1 + tmp3);
                            auto tmp5 = tmp1 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp1;
                            auto tmp7 = tmp6;
                            auto tmp8 = c10::convert<int64_t>(tmp7);
                            TORCH_CHECK((0 <= tmp8) & (tmp8 < 256LL), "index out of bounds: 0 <= tmp8 < 256LL");
                            auto tmp11 = decltype(tmp10)(tmp10 + tmp3);
                            auto tmp12 = tmp10 < 0;
                            auto tmp13 = tmp12 ? tmp11 : tmp10;
                            auto tmp14 = tmp13;
                            auto tmp15 = c10::convert<int64_t>(tmp14);
                            TORCH_CHECK((0 <= tmp15) & (tmp15 < 256LL), "index out of bounds: 0 <= tmp15 < 256LL");
                            auto tmp17 = in_ptr8[static_cast<int64_t>(tmp13 + (256LL*tmp6) + (65536LL*x2) + (131072LL*x0))];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                            auto tmp21 = decltype(tmp20)(-tmp20);
                            out_ptr45[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))] = tmp21;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16368LL); x0+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = static_cast<float>(1.0);
                    out_ptr46[static_cast<int64_t>(3LL*x0)] = tmp0;
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16368LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                {
                    auto tmp0 = in_ptr9[static_cast<int64_t>(x1 + (3LL*x0))];
                    auto tmp1 = in_ptr9[static_cast<int64_t>(3LL*x0)];
                    auto tmp3 = in_ptr9[static_cast<int64_t>(1LL + (3LL*x0))];
                    auto tmp6 = in_ptr9[static_cast<int64_t>(2LL + (3LL*x0))];
                    auto tmp2 = decltype(tmp1)(tmp1 * tmp1);
                    auto tmp4 = decltype(tmp3)(tmp3 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 + tmp4);
                    auto tmp7 = decltype(tmp6)(tmp6 * tmp6);
                    auto tmp8 = decltype(tmp5)(tmp5 + tmp7);
                    auto tmp9 = std::sqrt(tmp8);
                    auto tmp10 = static_cast<float>(1e-06);
                    auto tmp11 = max_propagate_nan(tmp9, tmp10);
                    auto tmp12 = tmp0 / tmp11;
                    out_ptr47[static_cast<int64_t>(x1 + (3LL*x0))] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_ptr10[static_cast<int64_t>(8LL*x0)];
                        auto tmp5 = in_ptr11[static_cast<int64_t>((3LL*x1) + (9LL*x0))];
                        auto tmp6 = in_ptr11[static_cast<int64_t>(9LL*x0)];
                        auto tmp8 = in_ptr11[static_cast<int64_t>(3LL + (9LL*x0))];
                        auto tmp11 = in_ptr11[static_cast<int64_t>(6LL + (9LL*x0))];
                        auto tmp19 = in_ptr10[static_cast<int64_t>(1LL + (8LL*x0))];
                        auto tmp23 = in_ptr10[static_cast<int64_t>(2LL + (8LL*x0))];
                        auto tmp27 = in_ptr10[static_cast<int64_t>(3LL + (8LL*x0))];
                        auto tmp1 = static_cast<float>(-1.0);
                        auto tmp2 = max_propagate_nan(tmp0, tmp1);
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = min_propagate_nan(tmp2, tmp3);
                        auto tmp7 = decltype(tmp6)(tmp6 * tmp6);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 + tmp9);
                        auto tmp12 = decltype(tmp11)(tmp11 * tmp11);
                        auto tmp13 = decltype(tmp10)(tmp10 + tmp12);
                        auto tmp14 = std::sqrt(tmp13);
                        auto tmp15 = static_cast<float>(1e-06);
                        auto tmp16 = max_propagate_nan(tmp14, tmp15);
                        auto tmp17 = tmp5 / tmp16;
                        auto tmp18 = decltype(tmp4)(tmp4 * tmp17);
                        auto tmp20 = max_propagate_nan(tmp19, tmp1);
                        auto tmp21 = min_propagate_nan(tmp20, tmp3);
                        auto tmp22 = decltype(tmp21)(tmp21 * tmp17);
                        auto tmp24 = max_propagate_nan(tmp23, tmp1);
                        auto tmp25 = min_propagate_nan(tmp24, tmp3);
                        auto tmp26 = decltype(tmp25)(tmp25 * tmp17);
                        auto tmp28 = max_propagate_nan(tmp27, tmp1);
                        auto tmp29 = min_propagate_nan(tmp28, tmp3);
                        auto tmp30 = decltype(tmp29)(tmp29 * tmp17);
                        out_ptr48[static_cast<int64_t>(x1 + (3LL*x0))] = tmp18;
                        out_ptr49[static_cast<int64_t>(x1 + (3LL*x0))] = tmp22;
                        out_ptr50[static_cast<int64_t>(x1 + (3LL*x0))] = tmp26;
                        out_ptr51[static_cast<int64_t>(x1 + (3LL*x0))] = tmp30;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(8LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1016LL); x1+=static_cast<int64_t>(8LL))
                        {
                            alignas(8) float tmp0[8*8];
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(in_out_ptr2 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL), tmp0, static_cast<int64_t>(8));
                            alignas(8) float tmp2[8*8];
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(out_ptr32 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL), tmp2, static_cast<int64_t>(8));
                            alignas(8) float tmp23[8*8];
                            alignas(8) float tmp35[8*8];
                            alignas(8) float tmp47[8*8];
                            alignas(8) float tmp59[8*8];
                            alignas(8) float tmp71[8*8];
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(out_ptr32 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL), tmp2, static_cast<int64_t>(8));
                            for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(8LL*x1_inner), static_cast<int64_t>(8));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<int64_t>(8LL*x1_inner), static_cast<int64_t>(8));
                                auto tmp8 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr43[static_cast<int64_t>((3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp9 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr47[static_cast<int64_t>((3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp11 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr43[static_cast<int64_t>(1LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp12 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr47[static_cast<int64_t>(1LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp15 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr43[static_cast<int64_t>(2LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp16 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr47[static_cast<int64_t>(2LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp24 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr48[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp27 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr48[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp31 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr48[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp36 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr49[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp39 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr49[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp43 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr49[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp48 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr50[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp51 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr50[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp55 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr50[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp60 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr51[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp63 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr51[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp67 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr51[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = static_cast<float>(30000.0);
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp10 = tmp8 * tmp9;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp14 = tmp10 + tmp13;
                                auto tmp17 = tmp15 * tmp16;
                                auto tmp18 = tmp14 + tmp17;
                                auto tmp19 = static_cast<float>(2852.367437761131);
                                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                                auto tmp21 = tmp18 * tmp20;
                                auto tmp22 = tmp7 + tmp21;
                                auto tmp25 = tmp24 - tmp8;
                                auto tmp26 = tmp25 * tmp9;
                                auto tmp28 = tmp27 - tmp11;
                                auto tmp29 = tmp28 * tmp12;
                                auto tmp30 = tmp26 + tmp29;
                                auto tmp32 = tmp31 - tmp15;
                                auto tmp33 = tmp32 * tmp16;
                                auto tmp34 = tmp30 + tmp33;
                                auto tmp37 = tmp36 - tmp8;
                                auto tmp38 = tmp37 * tmp9;
                                auto tmp40 = tmp39 - tmp11;
                                auto tmp41 = tmp40 * tmp12;
                                auto tmp42 = tmp38 + tmp41;
                                auto tmp44 = tmp43 - tmp15;
                                auto tmp45 = tmp44 * tmp16;
                                auto tmp46 = tmp42 + tmp45;
                                auto tmp49 = tmp48 - tmp8;
                                auto tmp50 = tmp49 * tmp9;
                                auto tmp52 = tmp51 - tmp11;
                                auto tmp53 = tmp52 * tmp12;
                                auto tmp54 = tmp50 + tmp53;
                                auto tmp56 = tmp55 - tmp15;
                                auto tmp57 = tmp56 * tmp16;
                                auto tmp58 = tmp54 + tmp57;
                                auto tmp61 = tmp60 - tmp8;
                                auto tmp62 = tmp61 * tmp9;
                                auto tmp64 = tmp63 - tmp11;
                                auto tmp65 = tmp64 * tmp12;
                                auto tmp66 = tmp62 + tmp65;
                                auto tmp68 = tmp67 - tmp15;
                                auto tmp69 = tmp68 * tmp16;
                                auto tmp70 = tmp66 + tmp69;
                                tmp22.store(tmp23 + static_cast<int64_t>(8LL*x1_inner));
                                tmp34.store(tmp35 + static_cast<int64_t>(8LL*x1_inner));
                                tmp46.store(tmp47 + static_cast<int64_t>(8LL*x1_inner));
                                tmp58.store(tmp59 + static_cast<int64_t>(8LL*x1_inner));
                                tmp70.store(tmp71 + static_cast<int64_t>(8LL*x1_inner));
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            }
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp23, static_cast<int64_t>(8), in_out_ptr2 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp35, static_cast<int64_t>(8), out_ptr52 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp47, static_cast<int64_t>(8), out_ptr53 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp59, static_cast<int64_t>(8), out_ptr54 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8)>(tmp71, static_cast<int64_t>(8), out_ptr55 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                        }
                        for(int64_t x1=static_cast<int64_t>(1016LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(7LL))
                        {
                            alignas(8) float tmp0[8*8];
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(7LL)>(in_out_ptr2 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL), tmp0, static_cast<int64_t>(8));
                            alignas(8) float tmp2[8*8];
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(7LL)>(out_ptr32 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL), tmp2, static_cast<int64_t>(8));
                            alignas(8) float tmp23[8*8];
                            alignas(8) float tmp35[8*8];
                            alignas(8) float tmp47[8*8];
                            alignas(8) float tmp59[8*8];
                            alignas(8) float tmp71[8*8];
                            at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(7LL)>(out_ptr32 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL), tmp2, static_cast<int64_t>(8));
                            for (long x1_inner = 0; x1_inner < static_cast<int64_t>(7LL); x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(8LL*x1_inner), static_cast<int64_t>(8));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<int64_t>(8LL*x1_inner), static_cast<int64_t>(8));
                                auto tmp8 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr43[static_cast<int64_t>((3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp9 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr47[static_cast<int64_t>((3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp11 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr43[static_cast<int64_t>(1LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp12 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr47[static_cast<int64_t>(1LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp15 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr43[static_cast<int64_t>(2LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp16 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr47[static_cast<int64_t>(2LL + (3LL*x1) + (3LL*x1_inner) + (3069LL*x0) + (3069LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp24 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr48[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp27 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr48[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp31 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr48[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp36 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr49[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp39 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr49[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp43 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr49[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp48 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr50[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp51 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr50[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp55 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr50[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp60 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr51[static_cast<int64_t>((3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp63 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr51[static_cast<int64_t>(1LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp67 =
                                [&]
                                {
                                    __at_align__ std::array<float, 8> tmpbuf;
                                    #pragma unroll 8
                                    for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                                    {
                                        tmpbuf[x0_inner] = out_ptr51[static_cast<int64_t>(2LL + (3LL*x0) + (3LL*x0_inner))];
                                    }
                                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = static_cast<float>(30000.0);
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp10 = tmp8 * tmp9;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp14 = tmp10 + tmp13;
                                auto tmp17 = tmp15 * tmp16;
                                auto tmp18 = tmp14 + tmp17;
                                auto tmp19 = static_cast<float>(2852.367437761131);
                                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                                auto tmp21 = tmp18 * tmp20;
                                auto tmp22 = tmp7 + tmp21;
                                auto tmp25 = tmp24 - tmp8;
                                auto tmp26 = tmp25 * tmp9;
                                auto tmp28 = tmp27 - tmp11;
                                auto tmp29 = tmp28 * tmp12;
                                auto tmp30 = tmp26 + tmp29;
                                auto tmp32 = tmp31 - tmp15;
                                auto tmp33 = tmp32 * tmp16;
                                auto tmp34 = tmp30 + tmp33;
                                auto tmp37 = tmp36 - tmp8;
                                auto tmp38 = tmp37 * tmp9;
                                auto tmp40 = tmp39 - tmp11;
                                auto tmp41 = tmp40 * tmp12;
                                auto tmp42 = tmp38 + tmp41;
                                auto tmp44 = tmp43 - tmp15;
                                auto tmp45 = tmp44 * tmp16;
                                auto tmp46 = tmp42 + tmp45;
                                auto tmp49 = tmp48 - tmp8;
                                auto tmp50 = tmp49 * tmp9;
                                auto tmp52 = tmp51 - tmp11;
                                auto tmp53 = tmp52 * tmp12;
                                auto tmp54 = tmp50 + tmp53;
                                auto tmp56 = tmp55 - tmp15;
                                auto tmp57 = tmp56 * tmp16;
                                auto tmp58 = tmp54 + tmp57;
                                auto tmp61 = tmp60 - tmp8;
                                auto tmp62 = tmp61 * tmp9;
                                auto tmp64 = tmp63 - tmp11;
                                auto tmp65 = tmp64 * tmp12;
                                auto tmp66 = tmp62 + tmp65;
                                auto tmp68 = tmp67 - tmp15;
                                auto tmp69 = tmp68 * tmp16;
                                auto tmp70 = tmp66 + tmp69;
                                tmp22.store(tmp23 + static_cast<int64_t>(8LL*x1_inner));
                                tmp34.store(tmp35 + static_cast<int64_t>(8LL*x1_inner));
                                tmp46.store(tmp47 + static_cast<int64_t>(8LL*x1_inner));
                                tmp58.store(tmp59 + static_cast<int64_t>(8LL*x1_inner));
                                tmp70.store(tmp71 + static_cast<int64_t>(8LL*x1_inner));
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            }
                            at::vec::transpose_mxn<float,static_cast<int64_t>(7LL),static_cast<int64_t>(8)>(tmp23, static_cast<int64_t>(8), in_out_ptr2 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(7LL),static_cast<int64_t>(8)>(tmp35, static_cast<int64_t>(8), out_ptr52 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(7LL),static_cast<int64_t>(8)>(tmp47, static_cast<int64_t>(8), out_ptr53 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(7LL),static_cast<int64_t>(8)>(tmp59, static_cast<int64_t>(8), out_ptr54 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                            at::vec::transpose_mxn<float,static_cast<int64_t>(7LL),static_cast<int64_t>(8)>(tmp71, static_cast<int64_t>(8), out_ptr55 + static_cast<int64_t>(x1 + (1023LL*x0)), static_cast<int64_t>(1023LL));
                        }
                        tmp_acc0_vec.store(out_ptr56 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_out_ptr2[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp1 = out_ptr47[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp4 = out_ptr32[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp6 = out_ptr56[static_cast<int64_t>(x0)];
                        auto tmp2 = decltype(tmp1)(-tmp1);
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp7 = static_cast<float>(1.0);
                        auto tmp8 = max_propagate_nan(tmp6, tmp7);
                        auto tmp9 = tmp5 / tmp8;
                        out_ptr57[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))] = tmp9;
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1023LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = out_ptr57[static_cast<int64_t>((3LL*x1) + (3069LL*x0))];
                        auto tmp2 = out_ptr57[static_cast<int64_t>(1LL + (3LL*x1) + (3069LL*x0))];
                        auto tmp5 = out_ptr57[static_cast<int64_t>(2LL + (3LL*x1) + (3069LL*x0))];
                        auto tmp11 = out_ptr48[static_cast<int64_t>(x2 + (3LL*x0))];
                        auto tmp12 = out_ptr43[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp14 = out_ptr52[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp15 = out_ptr47[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))];
                        auto tmp20 = in_ptr12[static_cast<int64_t>(x1)];
                        auto tmp22 = out_ptr49[static_cast<int64_t>(x2 + (3LL*x0))];
                        auto tmp24 = out_ptr53[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp29 = in_ptr12[static_cast<int64_t>(1023LL + x1)];
                        auto tmp32 = out_ptr50[static_cast<int64_t>(x2 + (3LL*x0))];
                        auto tmp34 = out_ptr54[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp39 = in_ptr12[static_cast<int64_t>(2046LL + x1)];
                        auto tmp42 = out_ptr51[static_cast<int64_t>(x2 + (3LL*x0))];
                        auto tmp44 = out_ptr55[static_cast<int64_t>(x1 + (1023LL*x0))];
                        auto tmp49 = in_ptr12[static_cast<int64_t>(3069LL + x1)];
                        auto tmp1 = decltype(tmp0)(tmp0 * tmp0);
                        auto tmp3 = decltype(tmp2)(tmp2 * tmp2);
                        auto tmp4 = decltype(tmp1)(tmp1 + tmp3);
                        auto tmp6 = decltype(tmp5)(tmp5 * tmp5);
                        auto tmp7 = decltype(tmp4)(tmp4 + tmp6);
                        auto tmp8 = std::sqrt(tmp7);
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp13 = decltype(tmp11)(tmp11 - tmp12);
                        auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                        auto tmp17 = decltype(tmp13)(tmp13 - tmp16);
                        auto tmp18 = std::tanh(tmp17);
                        auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 - tmp12);
                        auto tmp25 = decltype(tmp24)(tmp24 * tmp15);
                        auto tmp26 = decltype(tmp23)(tmp23 - tmp25);
                        auto tmp27 = std::tanh(tmp26);
                        auto tmp28 = decltype(tmp10)(tmp10 * tmp27);
                        auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                        auto tmp31 = decltype(tmp21)(tmp21 + tmp30);
                        auto tmp33 = decltype(tmp32)(tmp32 - tmp12);
                        auto tmp35 = decltype(tmp34)(tmp34 * tmp15);
                        auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                        auto tmp37 = std::tanh(tmp36);
                        auto tmp38 = decltype(tmp10)(tmp10 * tmp37);
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = decltype(tmp31)(tmp31 + tmp40);
                        auto tmp43 = decltype(tmp42)(tmp42 - tmp12);
                        auto tmp45 = decltype(tmp44)(tmp44 * tmp15);
                        auto tmp46 = decltype(tmp43)(tmp43 - tmp45);
                        auto tmp47 = std::tanh(tmp46);
                        auto tmp48 = decltype(tmp10)(tmp10 * tmp47);
                        auto tmp50 = decltype(tmp48)(tmp48 * tmp49);
                        auto tmp51 = decltype(tmp41)(tmp41 + tmp50);
                        in_out_ptr4[static_cast<int64_t>(x2 + (3LL*x1) + (3069LL*x0))] = tmp51;
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(1023LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp0 = out_ptr2[static_cast<int64_t>((3LL*x2) + (3069LL*x0) + (static_cast<int64_t>((1LL + x1)) % static_cast<int64_t>(3LL)))];
                            auto tmp1 = out_ptr57[static_cast<int64_t>((3LL*x2) + (3069LL*x0) + (static_cast<int64_t>((2LL + x1)) % static_cast<int64_t>(3LL)))];
                            auto tmp2 = in_out_ptr4[static_cast<int64_t>((3LL*x2) + (3069LL*x0) + (static_cast<int64_t>((2LL + x1)) % static_cast<int64_t>(3LL)))];
                            auto tmp5 = out_ptr2[static_cast<int64_t>((3LL*x2) + (3069LL*x0) + (static_cast<int64_t>((2LL + x1)) % static_cast<int64_t>(3LL)))];
                            auto tmp6 = out_ptr57[static_cast<int64_t>((3LL*x2) + (3069LL*x0) + (static_cast<int64_t>((1LL + x1)) % static_cast<int64_t>(3LL)))];
                            auto tmp7 = in_out_ptr4[static_cast<int64_t>((3LL*x2) + (3069LL*x0) + (static_cast<int64_t>((1LL + x1)) % static_cast<int64_t>(3LL)))];
                            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                            auto tmp10 = decltype(tmp4)(tmp4 - tmp9);
                            tmp_acc0 = tmp_acc0 + tmp10;
                        }
                        out_ptr59[static_cast<int64_t>(x1 + (3LL*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(48LL); x0+=static_cast<int64_t>(8LL))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr59 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(-500.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = at::vec::maximum(tmp0, tmp2);
                    auto tmp4 = static_cast<float>(500.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = at::vec::minimum(tmp3, tmp5);
                    tmp6.store(in_out_ptr5 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_copy_div_lift_fresh_mul_neg_sum_zeros_6 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(3LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(1023LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (3LL*x2) + (3069LL*x0)), static_cast<int64_t>(3LL));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + (3LL*x2) + (3069LL*x0)), static_cast<int64_t>(3LL));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp2, static_cast<int64_t>(3LL));
                        }
                        tmp_acc0_vec.store(in_out_ptr0 + static_cast<int64_t>(x1 + (3LL*x0)), static_cast<int64_t>(3LL));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + (3LL*x0))];
                        auto tmp1 = in_ptr3[static_cast<int64_t>(x1 + (3LL*x0))];
                        auto tmp32 = out_ptr0[static_cast<int64_t>(x1 + (3LL*x0))];
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = static_cast<float>(0.5);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 + tmp5);
                        auto tmp7 = decltype(tmp6)(tmp6 * tmp2);
                        auto tmp8 = static_cast<float>(2.0);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 + tmp9);
                        auto tmp11 = decltype(tmp7)(tmp7 * tmp4);
                        auto tmp12 = decltype(tmp1)(tmp1 + tmp11);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp2);
                        auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                        auto tmp15 = decltype(tmp10)(tmp10 + tmp14);
                        auto tmp16 = decltype(tmp1)(tmp1 + tmp13);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp2);
                        auto tmp18 = decltype(tmp15)(tmp15 + tmp17);
                        auto tmp19 = static_cast<float>(0.16666666666666666);
                        auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                        auto tmp21 = decltype(tmp0)(tmp0 + tmp20);
                        auto tmp22 = x1;
                        auto tmp23 = c10::convert<int64_t>(tmp22);
                        auto tmp24 = static_cast<int64_t>(1);
                        auto tmp25 = tmp23 < tmp24;
                        auto tmp26 = static_cast<int64_t>(2);
                        auto tmp27 = tmp23 < tmp26;
                        auto tmp28 = static_cast<float>(0.0);
                        auto tmp29 = static_cast<float>(-665.1179809570312);
                        auto tmp30 = tmp27 ? tmp28 : tmp29;
                        auto tmp31 = tmp25 ? tmp28 : tmp30;
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = static_cast<float>(0.014749262536873156);
                        auto tmp35 = decltype(tmp33)(tmp33 * tmp34);
                        auto tmp36 = decltype(tmp35)(tmp35 * tmp2);
                        auto tmp37 = decltype(tmp36)(tmp36 * tmp4);
                        auto tmp38 = decltype(tmp35)(tmp35 + tmp37);
                        auto tmp39 = decltype(tmp38)(tmp38 * tmp2);
                        auto tmp40 = decltype(tmp39)(tmp39 * tmp8);
                        auto tmp41 = decltype(tmp36)(tmp36 + tmp40);
                        auto tmp42 = decltype(tmp39)(tmp39 * tmp4);
                        auto tmp43 = decltype(tmp35)(tmp35 + tmp42);
                        auto tmp44 = decltype(tmp43)(tmp43 * tmp2);
                        auto tmp45 = decltype(tmp44)(tmp44 * tmp8);
                        auto tmp46 = decltype(tmp41)(tmp41 + tmp45);
                        auto tmp47 = decltype(tmp35)(tmp35 + tmp44);
                        auto tmp48 = decltype(tmp47)(tmp47 * tmp2);
                        auto tmp49 = decltype(tmp46)(tmp46 + tmp48);
                        auto tmp50 = decltype(tmp49)(tmp49 * tmp19);
                        auto tmp51 = decltype(tmp1)(tmp1 + tmp50);
                        out_ptr1[static_cast<int64_t>(x1 + (3LL*x0))] = tmp21;
                        in_out_ptr0[static_cast<int64_t>(x1 + (3LL*x0))] = tmp35;
                        out_ptr2[static_cast<int64_t>(x1 + (3LL*x0))] = tmp51;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(3LL); x0+=static_cast<int64_t>(3LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(8LL))
                    {
                        alignas(8) float tmp0[8*8];
                        at::vec::transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(3LL)>(in_ptr4 + static_cast<int64_t>(x0 + (3LL*x1)), static_cast<int64_t>(3LL), tmp0, static_cast<int64_t>(8));
                        alignas(8) float tmp27[8*8];
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(3LL); x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(8LL*x0_inner), static_cast<int64_t>(8));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + (16LL*x0) + (16LL*x0_inner)), static_cast<int64_t>(8));
                            auto tmp3 = static_cast<float>(0.01);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 * tmp4;
                            auto tmp6 = static_cast<float>(0.5);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp9 = tmp2 + tmp8;
                            auto tmp10 = tmp9 * tmp4;
                            auto tmp11 = static_cast<float>(2.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp14 = tmp5 + tmp13;
                            auto tmp15 = tmp10 * tmp7;
                            auto tmp16 = tmp2 + tmp15;
                            auto tmp17 = tmp16 * tmp4;
                            auto tmp18 = tmp17 * tmp12;
                            auto tmp19 = tmp14 + tmp18;
                            auto tmp20 = tmp2 + tmp17;
                            auto tmp21 = tmp20 * tmp4;
                            auto tmp22 = tmp19 + tmp21;
                            auto tmp23 = static_cast<float>(0.16666666666666666);
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp22 * tmp24;
                            auto tmp26 = tmp1 + tmp25;
                            tmp26.store(tmp27 + static_cast<int64_t>(8LL*x0_inner));
                        }
                        at::vec::transpose_mxn<float,static_cast<int64_t>(3LL),static_cast<int64_t>(8)>(tmp27, static_cast<int64_t>(8), out_ptr3 + static_cast<int64_t>(x0 + (3LL*x1)), static_cast<int64_t>(3LL));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp4 = out_ptr3[static_cast<int64_t>(1LL + (3LL*x0))];
                        auto tmp28 = out_ptr3[static_cast<int64_t>(2LL + (3LL*x0))];
                        auto tmp0 = x1;
                        auto tmp1 = c10::convert<int32_t>(tmp0);
                        auto tmp2 = static_cast<int32_t>(2);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = static_cast<float>(0.5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp9 = decltype(tmp4)(tmp4 + tmp8);
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = static_cast<float>(2.0);
                        auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 + tmp12);
                        auto tmp14 = decltype(tmp10)(tmp10 * tmp7);
                        auto tmp15 = decltype(tmp4)(tmp4 + tmp14);
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp5);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp11);
                        auto tmp18 = decltype(tmp13)(tmp13 + tmp17);
                        auto tmp19 = decltype(tmp4)(tmp4 + tmp16);
                        auto tmp20 = decltype(tmp19)(tmp19 * tmp5);
                        auto tmp21 = decltype(tmp18)(tmp18 + tmp20);
                        auto tmp22 = static_cast<float>(0.16666666666666666);
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = static_cast<int32_t>(0);
                        auto tmp25 = tmp24 == tmp24;
                        auto tmp26 = static_cast<int32_t>(1);
                        auto tmp27 = tmp1 == tmp26;
                        auto tmp29 = decltype(tmp28)(tmp28 * tmp5);
                        auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                        auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                        auto tmp32 = decltype(tmp31)(tmp31 * tmp5);
                        auto tmp33 = decltype(tmp32)(tmp32 * tmp11);
                        auto tmp34 = decltype(tmp29)(tmp29 + tmp33);
                        auto tmp35 = decltype(tmp32)(tmp32 * tmp7);
                        auto tmp36 = decltype(tmp28)(tmp28 + tmp35);
                        auto tmp37 = decltype(tmp36)(tmp36 * tmp5);
                        auto tmp38 = decltype(tmp37)(tmp37 * tmp11);
                        auto tmp39 = decltype(tmp34)(tmp34 + tmp38);
                        auto tmp40 = decltype(tmp28)(tmp28 + tmp37);
                        auto tmp41 = decltype(tmp40)(tmp40 * tmp5);
                        auto tmp42 = decltype(tmp39)(tmp39 + tmp41);
                        auto tmp43 = decltype(tmp42)(tmp42 * tmp22);
                        auto tmp44 = decltype(tmp43)(-tmp43);
                        auto tmp45 = static_cast<float>(0.0);
                        auto tmp46 = tmp27 ? tmp44 : tmp45;
                        auto tmp47 = tmp25 ? tmp46 : tmp45;
                        auto tmp48 = tmp3 ? tmp23 : tmp47;
                        out_ptr4[static_cast<int64_t>(x1 + (3LL*x0))] = tmp48;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp4 = out_ptr4[static_cast<int64_t>(x2 + (3LL*x0))];
                            auto tmp9 = out_ptr3[static_cast<int64_t>(2LL + (3LL*x0))];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int32_t>(tmp0);
                            auto tmp2 = static_cast<int32_t>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = x2;
                            auto tmp6 = c10::convert<int32_t>(tmp5);
                            auto tmp7 = static_cast<int32_t>(1);
                            auto tmp8 = tmp6 == tmp7;
                            auto tmp10 = static_cast<float>(0.01);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = static_cast<float>(0.5);
                            auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                            auto tmp14 = decltype(tmp9)(tmp9 + tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
                            auto tmp16 = static_cast<float>(2.0);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp11)(tmp11 + tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp12);
                            auto tmp20 = decltype(tmp9)(tmp9 + tmp19);
                            auto tmp21 = decltype(tmp20)(tmp20 * tmp10);
                            auto tmp22 = decltype(tmp21)(tmp21 * tmp16);
                            auto tmp23 = decltype(tmp18)(tmp18 + tmp22);
                            auto tmp24 = decltype(tmp9)(tmp9 + tmp21);
                            auto tmp25 = decltype(tmp24)(tmp24 * tmp10);
                            auto tmp26 = decltype(tmp23)(tmp23 + tmp25);
                            auto tmp27 = static_cast<float>(0.16666666666666666);
                            auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                            auto tmp29 = decltype(tmp28)(-tmp28);
                            auto tmp30 = static_cast<float>(0.0);
                            auto tmp31 = tmp8 ? tmp29 : tmp30;
                            auto tmp32 = tmp3 ? tmp31 : tmp30;
                            auto tmp33 = tmp3 ? tmp4 : tmp32;
                            out_ptr5[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp8 = out_ptr3[static_cast<int64_t>(3LL*x0)];
                            auto tmp29 = out_ptr5[static_cast<int64_t>(3LL + x2 + (9LL*x0))];
                            auto tmp31 = out_ptr5[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int32_t>(tmp0);
                            auto tmp2 = static_cast<int32_t>(1);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp4 = x2;
                            auto tmp5 = c10::convert<int32_t>(tmp4);
                            auto tmp6 = static_cast<int32_t>(2);
                            auto tmp7 = tmp5 == tmp6;
                            auto tmp9 = static_cast<float>(0.01);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = static_cast<float>(0.5);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = decltype(tmp8)(tmp8 + tmp12);
                            auto tmp14 = decltype(tmp13)(tmp13 * tmp9);
                            auto tmp15 = static_cast<float>(2.0);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = decltype(tmp10)(tmp10 + tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp11);
                            auto tmp19 = decltype(tmp8)(tmp8 + tmp18);
                            auto tmp20 = decltype(tmp19)(tmp19 * tmp9);
                            auto tmp21 = decltype(tmp20)(tmp20 * tmp15);
                            auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                            auto tmp23 = decltype(tmp8)(tmp8 + tmp20);
                            auto tmp24 = decltype(tmp23)(tmp23 * tmp9);
                            auto tmp25 = decltype(tmp22)(tmp22 + tmp24);
                            auto tmp26 = static_cast<float>(0.16666666666666666);
                            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                            auto tmp28 = decltype(tmp27)(-tmp27);
                            auto tmp30 = tmp7 ? tmp28 : tmp29;
                            auto tmp32 = tmp3 ? tmp30 : tmp31;
                            out_ptr6[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))] = tmp32;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        auto tmp4 = out_ptr3[static_cast<int64_t>(1LL + (3LL*x0))];
                        auto tmp28 = out_ptr3[static_cast<int64_t>(2LL + (3LL*x0))];
                        auto tmp44 = out_ptr6[static_cast<int64_t>(3LL + x1 + (9LL*x0))];
                        auto tmp46 = out_ptr6[static_cast<int64_t>(6LL + x1 + (9LL*x0))];
                        auto tmp0 = x1;
                        auto tmp1 = c10::convert<int32_t>(tmp0);
                        auto tmp2 = static_cast<int32_t>(0);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = static_cast<float>(0.5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp9 = decltype(tmp4)(tmp4 + tmp8);
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = static_cast<float>(2.0);
                        auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 + tmp12);
                        auto tmp14 = decltype(tmp10)(tmp10 * tmp7);
                        auto tmp15 = decltype(tmp4)(tmp4 + tmp14);
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp5);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp11);
                        auto tmp18 = decltype(tmp13)(tmp13 + tmp17);
                        auto tmp19 = decltype(tmp4)(tmp4 + tmp16);
                        auto tmp20 = decltype(tmp19)(tmp19 * tmp5);
                        auto tmp21 = decltype(tmp18)(tmp18 + tmp20);
                        auto tmp22 = static_cast<float>(0.16666666666666666);
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = decltype(tmp23)(-tmp23);
                        auto tmp25 = static_cast<int32_t>(2);
                        auto tmp26 = static_cast<int32_t>(1);
                        auto tmp27 = tmp25 == tmp26;
                        auto tmp29 = decltype(tmp28)(tmp28 * tmp5);
                        auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                        auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                        auto tmp32 = decltype(tmp31)(tmp31 * tmp5);
                        auto tmp33 = decltype(tmp32)(tmp32 * tmp11);
                        auto tmp34 = decltype(tmp29)(tmp29 + tmp33);
                        auto tmp35 = decltype(tmp32)(tmp32 * tmp7);
                        auto tmp36 = decltype(tmp28)(tmp28 + tmp35);
                        auto tmp37 = decltype(tmp36)(tmp36 * tmp5);
                        auto tmp38 = decltype(tmp37)(tmp37 * tmp11);
                        auto tmp39 = decltype(tmp34)(tmp34 + tmp38);
                        auto tmp40 = decltype(tmp28)(tmp28 + tmp37);
                        auto tmp41 = decltype(tmp40)(tmp40 * tmp5);
                        auto tmp42 = decltype(tmp39)(tmp39 + tmp41);
                        auto tmp43 = decltype(tmp42)(tmp42 * tmp22);
                        auto tmp45 = tmp3 ? tmp43 : tmp44;
                        auto tmp47 = tmp27 ? tmp45 : tmp46;
                        auto tmp48 = tmp3 ? tmp24 : tmp47;
                        out_ptr7[static_cast<int64_t>(x1 + (3LL*x0))] = tmp48;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp4 = out_ptr7[static_cast<int64_t>(x2 + (3LL*x0))];
                            auto tmp11 = out_ptr3[static_cast<int64_t>(2LL + (3LL*x0))];
                            auto tmp31 = out_ptr6[static_cast<int64_t>(3LL + x2 + (9LL*x0))];
                            auto tmp33 = out_ptr6[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int32_t>(tmp0);
                            auto tmp2 = static_cast<int32_t>(2);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<int32_t>(1);
                            auto tmp6 = tmp1 == tmp5;
                            auto tmp7 = x2;
                            auto tmp8 = c10::convert<int32_t>(tmp7);
                            auto tmp9 = static_cast<int32_t>(0);
                            auto tmp10 = tmp8 == tmp9;
                            auto tmp12 = static_cast<float>(0.01);
                            auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                            auto tmp14 = static_cast<float>(0.5);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp16 = decltype(tmp11)(tmp11 + tmp15);
                            auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                            auto tmp18 = static_cast<float>(2.0);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = decltype(tmp13)(tmp13 + tmp19);
                            auto tmp21 = decltype(tmp17)(tmp17 * tmp14);
                            auto tmp22 = decltype(tmp11)(tmp11 + tmp21);
                            auto tmp23 = decltype(tmp22)(tmp22 * tmp12);
                            auto tmp24 = decltype(tmp23)(tmp23 * tmp18);
                            auto tmp25 = decltype(tmp20)(tmp20 + tmp24);
                            auto tmp26 = decltype(tmp11)(tmp11 + tmp23);
                            auto tmp27 = decltype(tmp26)(tmp26 * tmp12);
                            auto tmp28 = decltype(tmp25)(tmp25 + tmp27);
                            auto tmp29 = static_cast<float>(0.16666666666666666);
                            auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                            auto tmp32 = tmp10 ? tmp30 : tmp31;
                            auto tmp34 = tmp6 ? tmp32 : tmp33;
                            auto tmp35 = tmp3 ? tmp4 : tmp34;
                            out_ptr8[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))] = tmp35;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
                    {
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                        {
                            auto tmp8 = out_ptr3[static_cast<int64_t>(3LL*x0)];
                            auto tmp28 = out_ptr8[static_cast<int64_t>(6LL + x2 + (9LL*x0))];
                            auto tmp30 = out_ptr8[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int32_t>(tmp0);
                            auto tmp2 = static_cast<int32_t>(2);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp4 = x2;
                            auto tmp5 = c10::convert<int32_t>(tmp4);
                            auto tmp6 = static_cast<int32_t>(1);
                            auto tmp7 = tmp5 == tmp6;
                            auto tmp9 = static_cast<float>(0.01);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = static_cast<float>(0.5);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = decltype(tmp8)(tmp8 + tmp12);
                            auto tmp14 = decltype(tmp13)(tmp13 * tmp9);
                            auto tmp15 = static_cast<float>(2.0);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = decltype(tmp10)(tmp10 + tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp11);
                            auto tmp19 = decltype(tmp8)(tmp8 + tmp18);
                            auto tmp20 = decltype(tmp19)(tmp19 * tmp9);
                            auto tmp21 = decltype(tmp20)(tmp20 * tmp15);
                            auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                            auto tmp23 = decltype(tmp8)(tmp8 + tmp20);
                            auto tmp24 = decltype(tmp23)(tmp23 * tmp9);
                            auto tmp25 = decltype(tmp22)(tmp22 + tmp24);
                            auto tmp26 = static_cast<float>(0.16666666666666666);
                            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                            auto tmp29 = tmp7 ? tmp27 : tmp28;
                            auto tmp31 = tmp3 ? tmp29 : tmp30;
                            out_ptr9[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))] = tmp31;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clamp_cos_div_linalg_vector_norm_mul_pow_rsub_sin_7 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(3LL*x0)];
                auto tmp21 = in_ptr0[static_cast<int64_t>(1LL + (3LL*x0))];
                auto tmp39 = in_ptr0[static_cast<int64_t>(2LL + (3LL*x0))];
                auto tmp1 = static_cast<float>(0.01);
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp3 = static_cast<float>(0.5);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 + tmp4);
                auto tmp6 = decltype(tmp5)(tmp5 * tmp1);
                auto tmp7 = static_cast<float>(2.0);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 + tmp8);
                auto tmp10 = decltype(tmp6)(tmp6 * tmp3);
                auto tmp11 = decltype(tmp0)(tmp0 + tmp10);
                auto tmp12 = decltype(tmp11)(tmp11 * tmp1);
                auto tmp13 = decltype(tmp12)(tmp12 * tmp7);
                auto tmp14 = decltype(tmp9)(tmp9 + tmp13);
                auto tmp15 = decltype(tmp0)(tmp0 + tmp12);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp1);
                auto tmp17 = decltype(tmp14)(tmp14 + tmp16);
                auto tmp18 = static_cast<float>(0.16666666666666666);
                auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                auto tmp20 = decltype(tmp19)(tmp19 * tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp1);
                auto tmp23 = decltype(tmp22)(tmp22 * tmp3);
                auto tmp24 = decltype(tmp21)(tmp21 + tmp23);
                auto tmp25 = decltype(tmp24)(tmp24 * tmp1);
                auto tmp26 = decltype(tmp25)(tmp25 * tmp7);
                auto tmp27 = decltype(tmp22)(tmp22 + tmp26);
                auto tmp28 = decltype(tmp25)(tmp25 * tmp3);
                auto tmp29 = decltype(tmp21)(tmp21 + tmp28);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp1);
                auto tmp31 = decltype(tmp30)(tmp30 * tmp7);
                auto tmp32 = decltype(tmp27)(tmp27 + tmp31);
                auto tmp33 = decltype(tmp21)(tmp21 + tmp30);
                auto tmp34 = decltype(tmp33)(tmp33 * tmp1);
                auto tmp35 = decltype(tmp32)(tmp32 + tmp34);
                auto tmp36 = decltype(tmp35)(tmp35 * tmp18);
                auto tmp37 = decltype(tmp36)(tmp36 * tmp36);
                auto tmp38 = decltype(tmp20)(tmp20 + tmp37);
                auto tmp40 = decltype(tmp39)(tmp39 * tmp1);
                auto tmp41 = decltype(tmp40)(tmp40 * tmp3);
                auto tmp42 = decltype(tmp39)(tmp39 + tmp41);
                auto tmp43 = decltype(tmp42)(tmp42 * tmp1);
                auto tmp44 = decltype(tmp43)(tmp43 * tmp7);
                auto tmp45 = decltype(tmp40)(tmp40 + tmp44);
                auto tmp46 = decltype(tmp43)(tmp43 * tmp3);
                auto tmp47 = decltype(tmp39)(tmp39 + tmp46);
                auto tmp48 = decltype(tmp47)(tmp47 * tmp1);
                auto tmp49 = decltype(tmp48)(tmp48 * tmp7);
                auto tmp50 = decltype(tmp45)(tmp45 + tmp49);
                auto tmp51 = decltype(tmp39)(tmp39 + tmp48);
                auto tmp52 = decltype(tmp51)(tmp51 * tmp1);
                auto tmp53 = decltype(tmp50)(tmp50 + tmp52);
                auto tmp54 = decltype(tmp53)(tmp53 * tmp18);
                auto tmp55 = decltype(tmp54)(tmp54 * tmp54);
                auto tmp56 = decltype(tmp38)(tmp38 + tmp55);
                out_ptr0[static_cast<int64_t>(x0)] = tmp56;
            }
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(3LL); x1+=static_cast<int64_t>(1LL))
            {
                for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(3LL); x2+=static_cast<int64_t>(1LL))
                {
                    auto tmp8 = out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp14 = in_out_ptr0[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))];
                    auto tmp21 = in_ptr1[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))];
                    auto tmp0 = x1;
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    auto tmp2 = x2;
                    auto tmp3 = c10::convert<int64_t>(tmp2);
                    auto tmp4 = tmp1 == tmp3;
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = tmp4 ? tmp5 : tmp6;
                    auto tmp9 = std::sqrt(tmp8);
                    auto tmp10 = static_cast<float>(1e-06);
                    auto tmp11 = max_propagate_nan(tmp9, tmp10);
                    auto tmp12 = std::sin(tmp11);
                    auto tmp13 = tmp12 / tmp11;
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp7)(tmp7 + tmp15);
                    auto tmp17 = std::cos(tmp11);
                    auto tmp18 = decltype(tmp5)(tmp5 - tmp17);
                    auto tmp19 = decltype(tmp11)(tmp11 * tmp11);
                    auto tmp20 = tmp18 / tmp19;
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = decltype(tmp16)(tmp16 + tmp22);
                    in_out_ptr0[static_cast<int64_t>(x2 + (3LL*x1) + (9LL*x0))] = tmp23;
                }
            }
        }
    }
}
''')


cpp_fused_add_clamp_div_mul_neg_8 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include "/var/folders/yj/_6b0pqvs5xg33v_xy5q2v9_40000gn/T/torchinductor_davidkorcak/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(4LL); x1+=static_cast<int64_t>(4LL))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(4LL + x1 + (8LL*x0)), static_cast<int64_t>(4LL));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(4LL));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + (4LL*x0)), static_cast<int64_t>(4LL));
                auto tmp30 =
                [&]
                {
                    __at_align__ std::array<float, 8> tmpbuf;
                    #pragma unroll 8
                    for (long x1_inner = 0; x1_inner < static_cast<int64_t>(4LL); x1_inner++)
                    {
                        tmpbuf[x1_inner] = in_ptr3[static_cast<int64_t>((2LL*x1) + (2LL*x1_inner))];
                    }
                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(4LL));
                }
                ()
                ;
                auto tmp32 =
                [&]
                {
                    __at_align__ std::array<float, 8> tmpbuf;
                    #pragma unroll 8
                    for (long x1_inner = 0; x1_inner < static_cast<int64_t>(4LL); x1_inner++)
                    {
                        tmpbuf[x1_inner] = in_ptr3[static_cast<int64_t>(1LL + (2LL*x1) + (2LL*x1_inner))];
                    }
                    return at::vec::Vectorized<float>::loadu(tmpbuf.data(), static_cast<int64_t>(4LL));
                }
                ()
                ;
                auto tmp2 = tmp1.neg();
                auto tmp3 = at::vec::maximum(tmp0, tmp2);
                auto tmp4 = at::vec::minimum(tmp3, tmp1);
                auto tmp6 = static_cast<float>(0.01);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp4 * tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp4 + tmp11;
                auto tmp13 = tmp12 * tmp7;
                auto tmp14 = static_cast<float>(2.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 * tmp15;
                auto tmp17 = tmp8 + tmp16;
                auto tmp18 = tmp13 * tmp10;
                auto tmp19 = tmp4 + tmp18;
                auto tmp20 = tmp19 * tmp7;
                auto tmp21 = tmp20 * tmp15;
                auto tmp22 = tmp17 + tmp21;
                auto tmp23 = tmp4 + tmp20;
                auto tmp24 = tmp23 * tmp7;
                auto tmp25 = tmp22 + tmp24;
                auto tmp26 = static_cast<float>(0.16666666666666666);
                auto tmp27 = at::vec::Vectorized<float>(tmp26);
                auto tmp28 = tmp25 * tmp27;
                auto tmp29 = tmp5 + tmp28;
                auto tmp31 = at::vec::maximum(tmp29, tmp30);
                auto tmp33 = at::vec::minimum(tmp31, tmp32);
                tmp4.store(out_ptr0 + static_cast<int64_t>(x1 + (4LL*x0)), static_cast<int64_t>(4LL));
                tmp33.store(out_ptr1 + static_cast<int64_t>(x1 + (4LL*x0)), static_cast<int64_t>(4LL));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 4), (4, 1))
    assert_size_stride(arg1_1, (16, 1023, 3), (3069, 3, 1))
    assert_size_stride(arg2_1, (4, 3), (3, 1))
    assert_size_stride(arg3_1, (4, 1023), (1023, 1))
    assert_size_stride(arg4_1, (1023, ), (1, ))
    assert_size_stride(arg5_1, (16, 3), (3, 1))
    assert_size_stride(arg6_1, (16, 3, 3), (9, 3, 1))
    assert_size_stride(arg7_1, (1023, ), (1, ))
    assert_size_stride(arg8_1, (16, 256, 256), (65536, 256, 1))
    assert_size_stride(arg9_1, (16, 2, 256, 256), (131072, 65536, 256, 1))
    assert_size_stride(arg10_1, (16, 3), (3, 1))
    assert_size_stride(arg11_1, (16, 3), (3, 1))
    assert_size_stride(arg12_1, (16, 8), (8, 1))
    assert_size_stride(arg13_1, (4, ), (1, ))
    assert_size_stride(arg14_1, (2, 4), (1, 2))
    buf3 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf0 = reinterpret_tensor(buf3, (16, 1), (3, 1), 0)  # alias
    buf2 = reinterpret_tensor(buf3, (16, 1), (3, 1), 2)  # alias
    buf11 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf8 = reinterpret_tensor(buf11, (16, 1), (3, 1), 0)  # alias
    buf10 = reinterpret_tensor(buf11, (16, 1), (3, 1), 2)  # alias
    buf1 = reinterpret_tensor(buf3, (16, 1), (3, 1), 1)  # alias
    buf7 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf4 = reinterpret_tensor(buf7, (16, 1), (3, 1), 0)  # alias
    buf5 = reinterpret_tensor(buf7, (16, 1), (3, 1), 1)  # alias
    buf6 = reinterpret_tensor(buf7, (16, 1), (3, 1), 2)  # alias
    buf9 = reinterpret_tensor(buf11, (16, 1), (3, 1), 1)  # alias
    buf15 = empty_strided_cpu((16, 9), (9, 1), torch.float32)
    buf12 = reinterpret_tensor(buf15, (16, 3), (9, 1), 0)  # alias
    buf13 = reinterpret_tensor(buf15, (16, 3), (9, 1), 3)  # alias
    buf14 = reinterpret_tensor(buf15, (16, 3), (9, 1), 6)  # alias
    buf16 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    buf34 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    buf52 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    buf71 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    cpp_fused_cat_cos_neg_ones_like_sin_stack_sub_zeros_like_0(arg0_1, buf3, buf7, buf11, arg1_1, arg2_1, buf0, buf2, buf8, buf10, buf1, buf4, buf5, buf6, buf9, buf12, buf13, buf14, buf16, buf34, buf52, buf71)
    del buf0
    del buf1
    del buf10
    del buf12
    del buf13
    del buf14
    del buf2
    del buf4
    del buf5
    del buf6
    del buf8
    del buf9
    buf17 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [flippter_coord_system_pts, flippter_coord_system_pts_1], Original ATen: [aten.sub, aten.bmm]
    extern_kernels.bmm(buf16, reinterpret_tensor(buf15, (16, 3, 3), (9, 3, 1), 0), out=buf17)
    buf21 = buf7; del buf7  # reuse
    buf18 = reinterpret_tensor(buf21, (16, 1), (3, 1), 0)  # alias
    buf20 = reinterpret_tensor(buf21, (16, 1), (3, 1), 2)  # alias
    buf29 = buf3; del buf3  # reuse
    buf26 = reinterpret_tensor(buf29, (16, 1), (3, 1), 0)  # alias
    buf28 = reinterpret_tensor(buf29, (16, 1), (3, 1), 2)  # alias
    buf19 = reinterpret_tensor(buf21, (16, 1), (3, 1), 1)  # alias
    buf25 = buf11; del buf11  # reuse
    buf22 = reinterpret_tensor(buf25, (16, 1), (3, 1), 0)  # alias
    buf23 = reinterpret_tensor(buf25, (16, 1), (3, 1), 1)  # alias
    buf24 = reinterpret_tensor(buf25, (16, 1), (3, 1), 2)  # alias
    buf27 = reinterpret_tensor(buf29, (16, 1), (3, 1), 1)  # alias
    buf33 = buf15; del buf15  # reuse
    buf30 = reinterpret_tensor(buf33, (16, 3), (9, 1), 0)  # alias
    buf31 = reinterpret_tensor(buf33, (16, 3), (9, 1), 3)  # alias
    buf32 = reinterpret_tensor(buf33, (16, 3), (9, 1), 6)  # alias
    cpp_fused_cat_cos_neg_ones_like_sin_stack_zeros_like_1(arg0_1, buf21, buf25, buf29, buf18, buf20, buf26, buf28, buf19, buf22, buf23, buf24, buf27, buf30, buf31, buf32)
    del buf18
    del buf19
    del buf20
    del buf22
    del buf23
    del buf24
    del buf26
    del buf27
    del buf28
    del buf30
    del buf31
    del buf32
    buf35 = buf16; del buf16  # reuse
    # Topologically Sorted Source Nodes: [flippter_coord_system_pts_3, flippter_coord_system_pts_4], Original ATen: [aten.sub, aten.bmm]
    extern_kernels.bmm(buf34, reinterpret_tensor(buf33, (16, 3, 3), (9, 3, 1), 0), out=buf35)
    buf39 = buf29; del buf29  # reuse
    buf36 = reinterpret_tensor(buf39, (16, 1), (3, 1), 0)  # alias
    buf38 = reinterpret_tensor(buf39, (16, 1), (3, 1), 2)  # alias
    buf47 = buf25; del buf25  # reuse
    buf44 = reinterpret_tensor(buf47, (16, 1), (3, 1), 0)  # alias
    buf46 = reinterpret_tensor(buf47, (16, 1), (3, 1), 2)  # alias
    buf37 = reinterpret_tensor(buf39, (16, 1), (3, 1), 1)  # alias
    buf43 = buf21; del buf21  # reuse
    buf40 = reinterpret_tensor(buf43, (16, 1), (3, 1), 0)  # alias
    buf41 = reinterpret_tensor(buf43, (16, 1), (3, 1), 1)  # alias
    buf42 = reinterpret_tensor(buf43, (16, 1), (3, 1), 2)  # alias
    buf45 = reinterpret_tensor(buf47, (16, 1), (3, 1), 1)  # alias
    buf51 = buf33; del buf33  # reuse
    buf48 = reinterpret_tensor(buf51, (16, 3), (9, 1), 0)  # alias
    buf49 = reinterpret_tensor(buf51, (16, 3), (9, 1), 3)  # alias
    buf50 = reinterpret_tensor(buf51, (16, 3), (9, 1), 6)  # alias
    cpp_fused_cat_cos_neg_ones_like_sin_stack_zeros_like_2(arg0_1, buf39, buf43, buf47, buf36, buf38, buf44, buf46, buf37, buf40, buf41, buf42, buf45, buf48, buf49, buf50)
    del buf36
    del buf37
    del buf38
    del buf40
    del buf41
    del buf42
    del buf44
    del buf45
    del buf46
    del buf48
    del buf49
    del buf50
    buf53 = buf34; del buf34  # reuse
    # Topologically Sorted Source Nodes: [flippter_coord_system_pts_6, flippter_coord_system_pts_7], Original ATen: [aten.sub, aten.bmm]
    extern_kernels.bmm(buf52, reinterpret_tensor(buf51, (16, 3, 3), (9, 3, 1), 0), out=buf53)
    buf58 = buf47; del buf47  # reuse
    buf55 = reinterpret_tensor(buf58, (16, 1), (3, 1), 0)  # alias
    buf57 = reinterpret_tensor(buf58, (16, 1), (3, 1), 2)  # alias
    buf66 = buf43; del buf43  # reuse
    buf63 = reinterpret_tensor(buf66, (16, 1), (3, 1), 0)  # alias
    buf65 = reinterpret_tensor(buf66, (16, 1), (3, 1), 2)  # alias
    buf56 = reinterpret_tensor(buf58, (16, 1), (3, 1), 1)  # alias
    buf70 = buf51; del buf51  # reuse
    buf67 = reinterpret_tensor(buf70, (16, 3), (9, 1), 0)  # alias
    buf62 = buf39; del buf39  # reuse
    buf59 = reinterpret_tensor(buf62, (16, 1), (3, 1), 0)  # alias
    buf60 = reinterpret_tensor(buf62, (16, 1), (3, 1), 1)  # alias
    buf61 = reinterpret_tensor(buf62, (16, 1), (3, 1), 2)  # alias
    buf68 = reinterpret_tensor(buf70, (16, 3), (9, 1), 3)  # alias
    buf64 = reinterpret_tensor(buf66, (16, 1), (3, 1), 1)  # alias
    buf69 = reinterpret_tensor(buf70, (16, 3), (9, 1), 6)  # alias
    cpp_fused_cat_cos_neg_ones_like_sin_stack_zeros_like_3(arg0_1, buf58, buf62, buf66, buf55, buf57, buf63, buf65, buf56, buf67, buf59, buf60, buf61, buf68, buf64, buf69)
    del buf55
    del buf56
    del buf57
    del buf59
    del buf60
    del buf61
    del buf63
    del buf64
    del buf65
    del buf67
    del buf68
    del buf69
    buf72 = buf52; del buf52  # reuse
    # Topologically Sorted Source Nodes: [flippter_coord_system_pts_9, flippter_coord_system_pts_10], Original ATen: [aten.sub, aten.bmm]
    extern_kernels.bmm(buf71, reinterpret_tensor(buf70, (16, 3, 3), (9, 3, 1), 0), out=buf72)
    buf54 = buf17; del buf17  # reuse
    buf73 = buf54; del buf54  # reuse
    cpp_fused_add_mul_4(buf73, arg3_1, arg2_1, buf35, buf53, buf72, arg4_1, arg1_1)
    del arg1_1
    del arg2_1
    del arg4_1
    buf74 = buf72; del buf72  # reuse
    # Topologically Sorted Source Nodes: [bmm_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf73, reinterpret_tensor(arg6_1, (16, 3, 3), (9, 1, 3), 0), out=buf74)
    buf75 = buf74; del buf74  # reuse
    buf76 = buf66; del buf66  # reuse
    buf77 = empty_strided_cpu((), (), torch.float32)
    buf78 = buf76; del buf76  # reuse
    buf79 = buf53; del buf53  # reuse
    buf80 = empty_strided_cpu((16, ), (1, ), torch.float32)
    buf88 = empty_strided_cpu((16, ), (1, ), torch.float32)
    buf81 = empty_strided_cpu((16, ), (1, ), torch.float32)
    buf93 = empty_strided_cpu((16, ), (1, ), torch.float32)
    buf82 = empty_strided_cpu((16, ), (1, ), torch.float32)
    buf87 = empty_strided_cpu((16, ), (1, ), torch.float32)
    buf86 = buf62; del buf62  # reuse
    buf84 = reinterpret_tensor(buf86, (16, 1), (3, 1), 1)  # alias
    buf92 = buf58; del buf58  # reuse
    buf89 = reinterpret_tensor(buf92, (16, 1), (3, 1), 0)  # alias
    buf83 = reinterpret_tensor(buf86, (16, 1), (3, 1), 0)  # alias
    buf90 = reinterpret_tensor(buf92, (16, 1), (3, 1), 1)  # alias
    buf85 = reinterpret_tensor(buf86, (16, 1), (3, 1), 2)  # alias
    buf97 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf94 = reinterpret_tensor(buf97, (16, 1), (3, 1), 0)  # alias
    buf91 = reinterpret_tensor(buf92, (16, 1), (3, 1), 2)  # alias
    buf95 = reinterpret_tensor(buf97, (16, 1), (3, 1), 1)  # alias
    buf96 = reinterpret_tensor(buf97, (16, 1), (3, 1), 2)  # alias
    buf101 = buf70; del buf70  # reuse
    buf98 = reinterpret_tensor(buf101, (16, 3), (9, 1), 0)  # alias
    buf99 = reinterpret_tensor(buf101, (16, 3), (9, 1), 3)  # alias
    buf100 = reinterpret_tensor(buf101, (16, 3), (9, 1), 6)  # alias
    buf103 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf104 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf105 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf106 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf107 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf108 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf109 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf111 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf112 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf113 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf102 = empty_strided_cpu((16, 1, 1023, 1), (1023, 16368, 1, 16368), torch.float32)
    buf110 = buf102; del buf102  # reuse
    buf114 = reinterpret_tensor(buf110, (16, 1023, 1), (1023, 1, 16368), 0); del buf110  # reuse
    buf115 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf118 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf119 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf120 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf121 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf122 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf123 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf124 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf126 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf127 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.int64)
    buf128 = empty_strided_cpu((16, 1023, 1), (1023, 1, 1), torch.float32)
    buf116 = buf35; del buf35  # reuse
    buf117 = empty_strided_cpu((16, 2, 1023, 1), (2046, 1023, 1, 32736), torch.float32)
    buf125 = buf117; del buf117  # reuse
    buf131 = buf71; del buf71  # reuse
    buf129 = reinterpret_tensor(buf131, (16, 1023, 2), (3069, 3, 1), 0)  # alias
    buf130 = reinterpret_tensor(buf131, (16, 1023, 1), (3069, 3, 1), 2)  # alias
    buf132 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    buf136 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf138 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf141 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf144 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf133 = buf114; del buf114  # reuse
    buf137 = empty_strided_cpu((16, 1023, 1), (1023, 1, 16368), torch.float32)
    buf139 = empty_strided_cpu((16, 1023, 1), (1023, 1, 16368), torch.float32)
    buf142 = empty_strided_cpu((16, 1023, 1), (1023, 1, 16368), torch.float32)
    buf145 = empty_strided_cpu((16, 1023, 1), (1023, 1, 16368), torch.float32)
    buf134 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf135 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    buf140 = empty_strided_cpu((16, 1023, 3), (3069, 3, 1), torch.float32)
    buf143 = buf140; del buf140  # reuse
    buf146 = buf143; del buf143  # reuse
    buf147 = empty_strided_cpu((16, 3), (3, 1), torch.float32)
    buf148 = buf147; del buf147  # reuse
    cpp_fused__to_copy_add_all_bitwise_and_clamp_div_eq_grid_sampler_2d_le_linalg_cross_linalg_vector_norm_mul_neg_ones_stack_sub_sum_tanh_5(buf75, buf78, buf133, buf125, buf146, buf148, arg5_1, arg7_1, buf86, buf92, buf97, arg8_1, arg10_1, arg11_1, arg9_1, buf131, arg12_1, arg6_1, arg3_1, buf77, buf79, buf80, buf88, buf81, buf93, buf82, buf87, buf84, buf89, buf83, buf90, buf85, buf94, buf91, buf95, buf96, buf98, buf99, buf100, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf111, buf112, buf113, buf115, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf126, buf127, buf128, buf116, buf129, buf130, buf132, buf136, buf138, buf141, buf144, buf137, buf139, buf142, buf145, buf134, buf135)
    del arg3_1
    del arg7_1
    del arg8_1
    del arg9_1
    del buf103
    del buf104
    del buf105
    del buf106
    del buf107
    del buf108
    del buf109
    del buf111
    del buf112
    del buf113
    del buf116
    del buf118
    del buf119
    del buf120
    del buf121
    del buf122
    del buf123
    del buf124
    del buf125
    del buf126
    del buf127
    del buf128
    del buf129
    del buf130
    del buf131
    del buf133
    del buf134
    del buf136
    del buf137
    del buf139
    del buf142
    del buf145
    del buf77
    del buf80
    del buf81
    del buf82
    del buf83
    del buf84
    del buf85
    del buf87
    del buf88
    del buf89
    del buf90
    del buf91
    del buf94
    del buf95
    del buf96
    # Topologically Sorted Source Nodes: [linalg_solve_ex], Original ATen: [aten._linalg_solve_ex]
    buf149 = torch.ops.aten._linalg_solve_ex.default(reinterpret_tensor(buf101, (16, 3, 3), (9, 3, 1), 0), buf148)
    buf150 = buf149[0]
    del buf149
    buf155 = buf97; del buf97  # reuse
    buf154 = buf92; del buf92  # reuse
    buf156 = buf155; del buf155  # reuse
    buf157 = buf86; del buf86  # reuse
    buf158 = buf144; del buf144  # reuse
    buf160 = buf141; del buf141  # reuse
    buf161 = empty_strided_cpu((16, 3, 3), (9, 3, 1), torch.float32)
    buf162 = empty_strided_cpu((16, 3, 3), (9, 3, 1), torch.float32)
    buf163 = buf138; del buf138  # reuse
    buf164 = empty_strided_cpu((16, 3, 3), (9, 3, 1), torch.float32)
    buf165 = empty_strided_cpu((16, 3, 3), (9, 3, 1), torch.float32)
    cpp_fused_add_copy_div_lift_fresh_mul_neg_sum_zeros_6(buf156, buf135, buf146, arg5_1, arg10_1, arg11_1, buf150, buf154, buf157, buf158, buf160, buf161, buf162, buf163, buf164, buf165)
    del arg10_1
    del arg11_1
    del arg5_1
    del buf160
    del buf161
    del buf162
    del buf163
    buf166 = buf164; del buf164  # reuse
    # Topologically Sorted Source Nodes: [omega_skew_squared], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf165, buf165, out=buf166)
    buf159 = reinterpret_tensor(buf93, (16, 1), (1, 16), 0); del buf93  # reuse
    buf167 = buf165; del buf165  # reuse
    cpp_fused_add_clamp_cos_div_linalg_vector_norm_mul_pow_rsub_sin_7(buf167, buf158, buf166, buf159)
    del buf159
    buf168 = buf166; del buf166  # reuse
    # Topologically Sorted Source Nodes: [theta_expand, sin_4, sin_term, mul_71, add_38, cos_4, sub_14, pow_1, cos_term, mul_72, delta_R, value_10], Original ATen: [aten.clamp, aten.sin, aten.div, aten.mul, aten.add, aten.cos, aten.rsub, aten.pow, aten.bmm]
    extern_kernels.bmm(buf167, arg6_1, out=buf168)
    del arg6_1
    del buf167
    buf169 = empty_strided_cpu((16, 4), (4, 1), torch.float32)
    buf170 = empty_strided_cpu((16, 4), (4, 1), torch.float32)
    cpp_fused_add_clamp_div_mul_neg_8(arg12_1, arg13_1, arg0_1, arg14_1, buf169, buf170)
    del arg0_1
    del arg12_1
    del arg13_1
    del arg14_1
    return (buf156, buf150, buf169, buf135, buf146, buf115, buf132, buf73, buf75, buf148, buf78, buf79, reinterpret_tensor(buf101, (16, 3, 3), (9, 3, 1), 0), buf154, buf157, buf168, buf158, buf170, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 4), (4, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((16, 1023, 3), (3069, 3, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((4, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((4, 1023), (1023, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1023, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((16, 3, 3), (9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1023, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16, 256, 256), (65536, 256, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((16, 2, 256, 256), (131072, 65536, 256, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((16, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((2, 4), (1, 2), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
