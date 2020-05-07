// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System.Runtime.CompilerServices;
using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class MomentumKernel : BaseOptimizerKernel
    {
        private bool _initw;
        private Matrix _vDw;
        private bool _initb;
        private Matrix _vDb;

        /// <summary>
        /// Momentum Parameter
        /// </summary>
        public double Tao { get; set; }

        public MomentumKernel(Momentum settings) : base(settings)
        {
            _initw = true;
            _initb = true;

            Tao = settings.Tao;
        }

        public MomentumKernel(double tao = 0.9, double alpha = 0.001) : base(new Momentum(tao, alpha))
        {
            _initw = true;
            _initb = true;
        }

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            if (_initw)
            {
                _initw = false;
                _vDw = (Tao * (Alpha * (w.Hadamard(dJdW))) + ((1 - Tao) * dJdW));
            }
            else
            {
                _vDw = (Tao * _vDw) + ((1 - Tao) * dJdW);
            }
            return _vDw;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            if (_initb)
            {
                _initb = false;
                _vDb = (Tao * (Alpha * (b.Hadamard(dJdB))) + ((1 - Tao) * dJdB));
            }
            else
            {
                _vDb = (Tao * _vDb) + ((1 - Tao) * dJdB);
            }
            return _vDb;
        }

        public override EOptimizerType Type() => EOptimizerType.Momentum;
    }

    public sealed class Momentum : Utility.Optimizer
    {
        public double Tao { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Momentum;

        public Momentum(double tao = 0.9, double alpha = 0.001) : base(alpha)
        {
            Tao = tao;
        }
    }
}