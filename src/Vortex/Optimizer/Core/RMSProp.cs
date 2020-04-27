// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class RMSProp : Utility.BaseOptimizer
    {
        private double Power2(double x)
        {
            return x * x;
        }
        private double OneOver(double x)
        {
            return 1 / x;
        }

        private Matrix _sDw;
        private Matrix _sDb;

        public double Rho { get; set; }
        public double Epsilon { get; set; }

        public RMSProp(RMSPropSettings settings) : base(settings)
        {
            Rho = settings.Rho;
            Epsilon = settings.Epsilon;
        }
        
        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            if (_sDw != null)
            {
                _sDw = Rho * _sDw + (1 - Rho) * dJdW.Map(Power2);
                Matrix mat = _sDw.Map(Math.Sqrt);
                mat.InMap(OneOver);
                mat.InHadamard(dJdW);
                return Alpha * mat;
            }
            _sDw = (Alpha * (w.Hadamard(dJdW)));
            return _sDw;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            if (_sDb != null)
            {
                _sDb = Rho * _sDb + (1 - Rho) * dJdB.Map(Power2);
                Matrix mat = _sDb.Map(Math.Sqrt);
                mat.InMap(OneOver);
                mat.InHadamard(dJdB);
                return Alpha * mat;
            }
            _sDb = (Alpha * (b.Hadamard(dJdB)));
            return _sDb;
        }

        public override EOptimizerType Type() => EOptimizerType.RMSProp;
    }

    public sealed class RMSPropSettings : OptimizerSettings
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.RMSProp;

        public RMSPropSettings(double rho, double epsilon, double alpha = 0.001) : base(alpha)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
    }
}
