using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializers.Utility;
using static System.Math;

namespace Vortex.Initializers.Kernels
{
    public class LeCunUniformKernel : BaseInitializerKernel
    {
        public LeCunUniformKernel(LeCunUniform initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            Matrix mat = w.Duplicate();
            mat.InRandomize(Sqrt(3.0 / (w.Columns)));
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.LeCunUniform;
    }

    public class LeCunUniform : BaseInitializer
    {
        public LeCunUniform(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.LeCunUniform;
    }
}