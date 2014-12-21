#include "solve_qp.h"


Real Delta(const Vec& alpha, const Vec& b_minus_H_alpha, const Mat& H, int u, int v);


void SolveQP(std::vector<Vec>& a, std::vector<Real>& b, Real lambda, Real epsilon_tol,
             Vec& alpha)
{
    int n = b.size();
    Vec bVec(n);
    for(int i = 0;i<n;i++)
    {
        bVec[i] = b[i];
    }

    std::fill(alpha.begin(), alpha.end(), 1./n);

    Mat H(n, n);
    for(int i = 0;i<n;i++)
    {
        for(int j = 0;j<n;j++)
        {
            H(i,j) = inner_prod(a[i], a[j]);
        }
    }
    H = H / lambda;


    Vec b_minus_H_alpha = bVec - prod(H, alpha);

    while( *std::max_element(b_minus_H_alpha.begin(), b_minus_H_alpha.end()) -
           inner_prod(alpha, b_minus_H_alpha)
           >
           epsilon_tol*fabs( inner_prod(bVec,alpha) - 0.5*inner_prod(alpha, prod(H, alpha)) )
         )
    {
        Real maxValue = b_minus_H_alpha(0);
        int maxIdx = 0;
        for(int i = 0;i<n;i++)
        {
            Real m = b_minus_H_alpha(i);
            if(m>maxValue)
            {
                maxValue = m;
                maxIdx = i;
            }
        }

        int u = maxIdx;


        Real maxDelta = -1;
        int v = -1;
        for(int i = 0;i<n;i++)
        {
            if(alpha(i)>0 && b_minus_H_alpha(u)>b_minus_H_alpha(i))
            {
                Real d = Delta(alpha, b_minus_H_alpha, H, u, i);
                if(d>maxDelta)
                {
                    v = i;
                    maxDelta = d;
                }
            }
        }


        Vec betta = alpha;
        betta(v) = 0;
        betta(u) = alpha(u) + alpha(v);

        Real H_sum = H(u,u) - 2*H(u,v) + H(v,v);
        Real tau = std::min(1., (b_minus_H_alpha(u)-b_minus_H_alpha(v)) / (alpha(v)*H_sum));


        alpha = (1-tau)*alpha + tau*betta;
        b_minus_H_alpha = bVec-prod(H,alpha);

    }
}



Real Delta(const Vec& alpha, const Vec& b_minus_H_alpha, const Mat& H, int u, int v)
{
    Real H_sum = H(u,u) - 2*H(u,v) + H(v,v);

    if( b_minus_H_alpha(u)-b_minus_H_alpha(v) < alpha(v)*H_sum )
    {
        return pow(b_minus_H_alpha(u)-b_minus_H_alpha(v), 2) / (2*H_sum);
    }
    else
    {
        return alpha(v)*(b_minus_H_alpha(u)-b_minus_H_alpha(v)) - 0.5*pow(alpha(v), 2)*H_sum;
    }
}


