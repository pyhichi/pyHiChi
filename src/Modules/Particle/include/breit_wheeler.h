#pragma once
#include "Constants.h"
#include "FP.h"
#include <cstdint>

#include <cmath>

using namespace constants;
namespace pfc
{
    class Breit_wheeler
    {
    private:
        inline static const uint64_t int_g_pair[] = {
#include "QED_pair_half.in" 
        };

        double* g_pair;
        FP preFactor;

        //Corless R. M., Gonnet G. H., Hare D. E., Jeffrey D. J., & Knuth D. E.,
        //On the LambertW function. (1996).
        //Advances in Computational mathematics, 5(1), 329-359
        forceinline FP lambertW(FP x) {
            FP w = x;
            if (x > 1.8)
            {
                FP lnx = log(x);
                FP lnlnx = log(lnx);
                w = lnx - lnlnx + lnlnx / lnx;
            }
#pragma unroll(25)
            for (int i = 0; i < 25; i++) {
                FP expw = exp(w);
                FP wexpw = w * expw;
                FP w1expw = (w + 1) * expw;
                FP wexpwmx = (wexpw - x);
                w -= wexpwmx / (w1expw - (w + 2) * wexpwmx / (2 * w + 2));
                if (abs((x - wexpw) / w1expw) < 1e-12)
                    break;
            }
            return w;
        }
    public:
        Breit_wheeler() {
            g_pair = (double*)int_g_pair;

            preFactor = sqr(Constants<FP>::electronCharge()) * Constants<FP>::electronMass()
                * Constants<FP>::lightVelocity() / sqr(Constants<FP>::planck());

            preFactor *= sqrt((FP)3) / ((FP)2.0 * Constants<FP>::pi());
        }

        forceinline FP rate(const FP& chi)
        {
            FP a = ((FP)2.0) / ((FP)3.0 * chi);

            FP integral1 = 0;
            FP integral2 = 0;

            FP b = 4 * a;

            if (b < 1e-6)
            {
                FP x = pow(b, 1.0 / 3.0);
                integral1 = 1.5888757318580334808 + (4.621242391684 * 1e-9 +
                    (-1.8138014771310675045 + 0.000258334152383
                        * x) * x) * x;
                integral1 *= x;
            }
            else if (b < 1e-1)
            {
                FP x = pow(b, 1.0 / 3.0);
                integral1 = 1.58887575649 + (-2.88997830078 * 1e-6 +
                    (-1.81369737480 + (-0.00159621376721 +
                        (0.0131013808926 + (0.881783275752 +
                            (-0.100442434842 - 0.224972034631
                                * x) * x) * x) * x) * x) * x) * x;
                integral1 *= x;
            }
            else if (b < 1)
            {
                FP x = pow(b, 1.0 / 3.0);
                integral1 = 0.00222274692761579 + (1.55966451731044 +
                    (0.169055775710587 + (-2.37812332098245 +
                        (2.77990824833067 + (-1.62549030970 +
                            (0.507039151066 - 0.0676910228287
                                * x) * x) * x) * x) * x) * x) * x;
                integral1 *= exp(-b);
            }
            else if (b < 4)
            {

                FP x = pow(b, 1.0 / 3.0);
                integral1 = -0.0863337964364 + (2.12727019609 +
                    (-1.40279492272 + (0.0617954715806 +
                        (0.485497729909 + (-0.317700617716 +
                            (0.0885304524273 - 0.00967875703875
                                * x) * x) * x) * x) * x) * x) * x;
                integral1 *= exp(-b);
            }
            else if (b < 200)
            {
                FP x = 1.0 / b;
                integral1 = 1.11072184643 + (-0.262362340096 +
                    (0.143672779384 + (0.160152626879 +
                        (-1.40283063884 + (4.65809948572 +
                            (-8.51231822514 + 6.75456138684
                                * x) * x) * x) * x) * x) * x) * x;
                integral1 *= exp(-b);
            }
            else
                integral1 = 1.11072184643 * exp(-b);


            if (a < 0.015625) //2^-6
            {
                FP a1_3 = pow(a, 1.0 / 3.0);
                FP inv_a = 1.0 / a;
                FP a2 = a * a;
                FP a2_3 = a1_3 * a1_3;
                integral2 = 26.4383381757 + (-9.18235928145 +
                    (22.6194671058 + (-5.44139809271 +
                        (2.20691013218 * a1_3
                            - 6.70820546145 * a * a2_3
                            - 8.27591299568 * a2 * a1_3
                            - 9.15670045488 * a2 * a * a2_3)
                        * inv_a) * inv_a) * inv_a) * inv_a;
                integral2 *= a2 * a2;
            }
            else if (a < 8)
            {
                FP integral2_1, integral2_2;
                if (a < 0.5) //integral2_1
                {
                    FP x = pow(a, 1.0 / 3.0);
                    integral2_1 = 1.57027828224594 + (-1.08877277807399 +
                        (-0.179460809334361 + (4.80324684519270 +
                            (-9.74348350227989 + (9.85936580010627 +
                                (-5.290476246 + 1.20361312
                                    * x) * x) * x) * x) * x) * x) * x;
                    integral2_1 *= exp(-4 * a);
                }
                else
                {
                    FP x = pow(a, 1.0 / 6.0);
                    integral2_1 = -1.20066778173054 + (10.2368863985165 +
                        (-25.1115805339200 + (35.9639393894849 +
                            (-29.9577636462431 + (14.7275473913076 +
                                (-3.98478567702 + 0.459485809949
                                    * x) * x) * x) * x) * x) * x) * x;
                    integral2_1 *= exp(-4 * a) / x;
                }

                if (a < 0.125) //integral2_2
                {
                    FP x = pow(a, -1.0 / 6.0);
                    integral2_2 = 4.55073745530463 + (-18.9753607092870 +
                        (32.3758110391415 + (-29.1915770298345 +
                            (15.3907251618537 + (-4.81171065623 +
                                (0.832012023685 - 0.0616171211135
                                    * x) * x) * x) * x) * x) * x) * x;
                    integral2_2 *= exp(-4 * a) * pow(a, -1.0 / 3.0);
                }
                else if (a < 1)
                {
                    FP x = pow(a, -1.0 / 4.0);
                    integral2_2 = -0.311884654115626 + (1.80139927643145 +
                        (-4.30266204865178 + (5.37732685991004 +
                            (-3.50874100787307 + (1.282451108030 +
                                (-0.2517103475662 + 0.02079270065
                                    * x) * x) * x) * x) * x) * x) * x;
                    integral2_2 *= exp(-4 * a);
                }
                else
                {
                    FP x = pow(a, 1.0 / 6.0);
                    integral2_2 = 2.71429586590 + (-16.1475601299 +
                        (39.5087568627 + (-51.5369108153 +
                            (39.5252842521 + (-17.9992925987 +
                                (4.52979557008 - 0.487397088117
                                    * x) * x) * x) * x) * x) * x) * x;
                    integral2_2 *= exp(-4 * a) / a;
                }

                integral2 = integral2_1 - integral2_2;
            }
            else if (a < 50)
            {
                FP x = 1 / a;
                integral2 = 1.11072073452 + (-0.111843403492 +
                    (0.0422690451294 + (-0.0268525026114 +
                        (0.023559419306 + (-0.0241254152815 +
                            (0.0187749900469 + 0.00126008980415
                                * x) * x) * x) * x) * x) * x) * x;
                integral2 *= exp(-4 * a);
            }
            else
                integral2 = 1.11072073452 * exp(-4 * a);


            return preFactor * std::max(-integral1 * 0.25 + integral2, 0.0);
        }

        forceinline FP inv_cdf(FP r, FP chi)
        {
            double* g = g_pair;
            int N = 128;
            FP a = ((FP)2.0) / ((FP)3.0 * chi);
            FP x_target = a;

            FP logA = log(a) / log((FP)1.5);

            int index_a = std::max(std::min((int)std::floor(logA), 29), -30);

            FP delta;
            int index_f = 0;
            FP f1, f2;
            if (r >= 0.05)
            {
                index_f = std::floor((0.5 - r) / 0.45 * 64.0);
                f2 = 0.5 - 0.45 / 64.0 * index_f;
                f1 = f2 - 0.45 / 64.0;
                index_f = 126 - index_f;
            }
            else if (r > 0.0)
            {
                index_f = (62 - (int)std::floor(-log2(r / 0.05) / 0.2));
                if (index_f > 0)
                {
                    f2 = 0.05 * pow(2.0, -(62 - index_f) * 0.2);
                    f1 = 0.05 * pow(2.0, -(63 - index_f) * 0.2);
                }
                else
                {
                    index_f = 0;
                    f2 = 0.05 * pow(2.0, -(62) * 0.2);
                    f1 = 0.0;
                }
            }
            else
            {
                return 0.0;
            }


            index_f += (index_a + 30) * N;

            FP x1 = pow(1.5, (FP)(-index_a));         // !!!
            FP x2 = pow(1.5, (FP)(-(index_a + 1)));   // !!!
            x_target = 1.0 / x_target;

            FP y0 = 0.05 * pow(2.0, -(62) * 0.2);
            if (r > y0)
            {
                FP z1 = g[index_f] + (g[index_f + 1] - g[index_f]) * (r - f1) / (f2 - f1);
                index_f += N;
                FP z2 = g[index_f] + (g[index_f + 1] - g[index_f]) * (r - f1) / (f2 - f1);

                delta = z1 + (z2 - z1) * (x_target - x1) / (x2 - x1);
            }
            else
            {
                FP z0 = (x2 - x_target) / (x2 - x1) * g[index_f + 1]
                    + (x_target - x1) / (x2 - x1) * g[index_f + 1 + N];
                FP x0 = z0 * chi;
                FP coeff = y0 * (pow(x0, (FP)-1.5) * exp(((FP)2.0)/((FP)3.0 * x0)));
                FP lw = lambertW(((FP)4.0) / ((FP)9.0) * pow(coeff / r, ((FP)2.0) / ((FP)3.0)));

                delta = ((FP)4.0)/((FP)9.0 * chi * lw);
            }
            return delta;
        }
    };
}