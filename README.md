float hh = 0,vv = 0,hv = 0;
float diffh,diffv;

int offset;

int maxhv = 0,maxi,maxj;

float e1,e2;
float s1,s2;
float theda;
float temp,temp1,temp2;
float sigma;

double demda = 0.0000001;

float delta;

float *pKernel;
float d;
float scale;

int   diaW = 2*radius_w +1;
int   diah = 2*radius_h +1;
float *pKernelL;

int idx;

int gradient_w = MIN(radius_w,3);
int gradient_h = MIN(radius_h,3);

int pos;


pKernel = pSteerkernel;

pSrcL = pSrc;

for(i = -gradient_h; i <= gradient_h;i++)
{
    for(j = -gradient_w; j <= gradient_w;j++)
    {
        pos   = i * stride + j;
        diffh = pSrcL[pos+1] - pSrcL[pos - 1];
        diffv = pSrcL[pos+stride] - pSrcL[pos-stride];

        hh += diffh * diffh;
        vv += diffv * diffv;
        hv += diffh * diffv;

        if(maxhv < diffh * diffv) //不予绝对值比较，只记录两个方向梯度同符号的最大值
        {
            maxhv = diffh * diffv;
            maxi = i;
            maxj = j;
        }
    }
}
#if 0 /* **减去最大的cross误差，避免一些冲击早上干扰 */ if(maxhv > 0) { diffh = pSrcL[maxi * stride + maxj] - pSrcL[maxi * stride + maxj - 1]; diffv = pSrcL[maxi * stride + maxj] - pSrcL[(maxi - 1) * stride + maxj ];

    hh -= diffh * diffh;
    vv -= diffv * diffv;
    hv -= diffh * diffv;
}
#endif

hh /= (255*255);
vv /= (255*255);
hv /= (255*255);

if(0 == hv)
{
    if(hh > vv)
    {
        e1 = hh;
        e2 = vv;
        theda = PI/2;
    }
    else
    {
        e1 = vv;
        e2 = hh;
        theda = 0;
    }
}
else
{
    temp1 = hh + vv;
    temp2 = sqrt((hh + vv)*(hh + vv) - 4*(hh * vv - hv*hv));
    e1 = (temp1 + temp2)/2;
    e2 = (temp1 - temp2)/2;

    if(e1 < e2)
    {
        temp = e1;
        e1 = e2;
        e2 = temp;
    }

    theda = atan((e2-vv)/hv);
}

s1 = sqrt(e1);
s2 = sqrt(e2);

delta = sqrt((s1 + demda)/(s2 + demda));

sigma = sqrt((s1*s2 + demda)/((diaW)*(diah)));//差值的数目为((diaW-1)*(diah-1))

scale = 2 * sigma;

pKernelL = pKernel;
idx = 0;
for(i = -radius_h; i <= radius_h;i++)
{
    for(j = -radius_w; j <= radius_w;j++)
    {
        float s = sin(theda) ;
        float c = cos(theda) ;

        temp1 = j * s + i* c;
        temp2 = j * c - i * s;

        d =  1/delta * temp1 * temp1 + delta * temp2 * temp2;
        d *= sigma;
        
        //pKernelL[idx] = sqrt(scale) * exp(-d/h);
        pKernelL[idx] = exp(-d/h);
        idx++;
    }

}
}
