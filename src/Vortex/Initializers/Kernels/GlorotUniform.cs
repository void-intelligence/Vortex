using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializers.Utility;
using static System.Math;

namespace Vortex.Initializers.Kernels
{
    public class GlorotUniformKernel : BaseInitializerKernel
    {
        public GlorotUniformKernel(GlorotUniform initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            Matrix mat = w.Duplicate();
            mat.InRandomize(Sqrt(6.0 / (w.Columns + w.Rows)));
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.GlorotUniform;
    }

    public class GlorotUniform : BaseInitializer
    {
        public GlorotUniform(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.GlorotUniform;
    }
}