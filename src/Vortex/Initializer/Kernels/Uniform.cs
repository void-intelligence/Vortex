using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;

namespace Vortex.Initializer.Kernels
{
    public class UniformKernel : BaseInitializerKernel
    {
        public UniformKernel(Uniform initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            Matrix mat = w.Duplicate();
            mat.InRandomize(Min, Max, EDistribution.Uniform);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.Uniform;
    }

    public class Uniform : BaseInitializer
    {
        public Uniform(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.Uniform;
    }
}