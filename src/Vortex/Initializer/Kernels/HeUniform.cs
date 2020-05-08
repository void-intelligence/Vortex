using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public class HeUniformKernel : BaseInitializerKernel
    {
        public HeUniformKernel(HeUniform initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            Matrix mat = w.Duplicate();
            mat.InRandomize(Sqrt(6.0 / w.Columns));
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.HeUniform;
    }

    public class HeUniform : BaseInitializer
    {
        public HeUniform(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.HeUniform;
    }
}