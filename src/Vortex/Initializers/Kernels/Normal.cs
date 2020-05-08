using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializers.Utility;

namespace Vortex.Initializers.Kernels
{
    public class NormalKernel : BaseInitializerKernel
    {
        public NormalKernel(Normal initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            Matrix mat = w.Duplicate();
            mat.InRandomize(Min, Max, EDistribution.Gaussian);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.Normal;
    }

    public class Normal : BaseInitializer
    {
        public Normal(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.Normal;
    }
}