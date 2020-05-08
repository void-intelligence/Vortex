using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializers.Utility;
using static System.Math;

namespace Vortex.Initializers.Kernels
{
    public class GlorotNormalKernel : BaseInitializerKernel
    {
        private int _denom;
        private double Method(double input)
        {
            return input * Sqrt(2.0 / (_denom));
        }

        public GlorotNormalKernel(GlorotNormal initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            _denom = w.Columns + w.Rows;

            Matrix mat = w.Duplicate();
            mat.InRandomize(Min,Max,EDistribution.Gaussian);
            mat.InMap(Method);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.GlorotNormal;
    }

    public class GlorotNormal : BaseInitializer
    {
        public GlorotNormal(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.GlorotNormal;
    }
}