using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public class HeNormalKernel : BaseInitializerKernel
    {
        private int _h;
        private double Method(double input)
        {
            return input * Sqrt(2.0 / _h);
        }

        public HeNormalKernel(HeNormal initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            _h = w.Columns;
            Matrix mat = w.Duplicate();
            mat.InRandomize(Min, Max, EDistribution.Gaussian);
            mat.InMap(Method);
            mat *= Scale;
            return mat;
        }
        public override EInitializerType Type() => EInitializerType.HeNormal;
    }

    public class HeNormal : BaseInitializer
    {
        public HeNormal(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.HeNormal;
    }
}