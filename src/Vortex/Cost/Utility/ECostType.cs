// Copyright © 2020 Void-Intelligence All Rights Reserved.

namespace Vortex.Cost.Utility
{
    public enum ECostType
    {
        BinaryCrossEntropy,
        BinaryExponential,
        BinaryHinge,
        BinaryHingeSquared,
        BinaryLogit,
        CategoricalCorssEntropy,
        CategoricalKLD,
        LegacyGeneralizedKullbackLeibler,
        LegacyHellingerDistance,
        LegacyItakuraSaitoDistance,
        LegacyKullbackLeiblerDivergance,
        LegacyQuadraticCost,
        RegressionCosineProximity,
        RegressionHuber,
        RegressionLogCosh,
        RegressionMae,
        RegressionMAPE,
        RegressionMSE,
        RegressionMSLE,
        RegressionPoisson,
        RegressionQuantile
    }
}
