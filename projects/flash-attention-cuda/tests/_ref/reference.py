"""Reference for flash-attention-cuda.

Each step's CUDA source is a module-level string named EXACTLY for the step, and
the CUDA host wrapper inside it is also named for the step. So ns[name] is the
learner's source when grading and this reference's when verifying, and the graded
function is always mod.<name>. grade(ns, name) compiles + checks vs a torch
oracle; it no-ops without a GPU so CI/verify still pass on CPU-only machines.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
from cuda_harness import compile_cuda, have_gpu  # noqa: E402

H = '#include <torch/extension.h>\n#include <cuda_runtime.h>\n'

# ------------------------------- Part 1: primitives -------------------------------
vector_add = H + r'''
__global__ void va_k(const float* a,const float* b,float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=a[i]+b[i];}
torch::Tensor vector_add(torch::Tensor a,torch::Tensor b){auto o=torch::empty_like(a);int n=a.numel();int t=256,bl=(n+t-1)/t;
va_k<<<bl,t>>>(a.data_ptr<float>(),b.data_ptr<float>(),o.data_ptr<float>(),n);return o;}'''

scale_array = H + r'''
__global__ void sca_k(const float* a,float s,float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=a[i]*s;}
torch::Tensor scale_array(torch::Tensor a,double s){auto o=torch::empty_like(a);int n=a.numel();int t=256,bl=(n+t-1)/t;
sca_k<<<bl,t>>>(a.data_ptr<float>(),(float)s,o.data_ptr<float>(),n);return o;}'''

elementwise_exp = H + r'''
__global__ void exp_k(const float* a,float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=expf(a[i]);}
torch::Tensor elementwise_exp(torch::Tensor a){auto o=torch::empty_like(a);int n=a.numel();int t=256,bl=(n+t-1)/t;
exp_k<<<bl,t>>>(a.data_ptr<float>(),o.data_ptr<float>(),n);return o;}'''

row_max = H + r'''
__global__ void rmax_k(const float* a,float* o,int R,int C){int r=blockIdx.x;if(r>=R)return;float m=-1e30f;
for(int c=threadIdx.x;c<C;c+=blockDim.x)m=fmaxf(m,a[r*C+c]);__shared__ float sm[256];sm[threadIdx.x]=m;__syncthreads();
for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+s]);__syncthreads();}
if(threadIdx.x==0)o[r]=sm[0];}
torch::Tensor row_max(torch::Tensor a){int R=a.size(0),C=a.size(1);auto o=torch::empty({R},a.options());
rmax_k<<<R,256>>>(a.data_ptr<float>(),o.data_ptr<float>(),R,C);return o;}'''

row_sum = H + r'''
__global__ void rsum_k(const float* a,float* o,int R,int C){int r=blockIdx.x;if(r>=R)return;float v=0.f;
for(int c=threadIdx.x;c<C;c+=blockDim.x)v+=a[r*C+c];__shared__ float sm[256];sm[threadIdx.x]=v;__syncthreads();
for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sm[threadIdx.x]+=sm[threadIdx.x+s];__syncthreads();}
if(threadIdx.x==0)o[r]=sm[0];}
torch::Tensor row_sum(torch::Tensor a){int R=a.size(0),C=a.size(1);auto o=torch::empty({R},a.options());
rsum_k<<<R,256>>>(a.data_ptr<float>(),o.data_ptr<float>(),R,C);return o;}'''

# ------------------------------- Part 2: matrix ops -------------------------------
dot_product = H + r'''
__global__ void dot_k(const float* a,const float* b,float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;
__shared__ float sm[256];float v=(i<n)?a[i]*b[i]:0.f;sm[threadIdx.x]=v;__syncthreads();
for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sm[threadIdx.x]+=sm[threadIdx.x+s];__syncthreads();}
if(threadIdx.x==0)atomicAdd(o,sm[0]);}
torch::Tensor dot_product(torch::Tensor a,torch::Tensor b){auto o=torch::zeros({},a.options());int n=a.numel();int t=256,bl=(n+t-1)/t;
dot_k<<<bl,t>>>(a.data_ptr<float>(),b.data_ptr<float>(),o.data_ptr<float>(),n);return o;}'''

matmul = H + r'''
__global__ void mm_k(const float* A,const float* B,float* C,int M,int K,int N){
int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<M&&c<N){float s=0;for(int k=0;k<K;k++)s+=A[r*K+k]*B[k*N+c];C[r*N+c]=s;}}
torch::Tensor matmul(torch::Tensor A,torch::Tensor B){int M=A.size(0),K=A.size(1),N=B.size(1);auto C=torch::empty({M,N},A.options());
dim3 t(16,16),bl((N+15)/16,(M+15)/16);mm_k<<<bl,t>>>(A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,K,N);return C;}'''

transpose = H + r'''
__global__ void tr_k(const float* A,float* O,int M,int N){int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<M&&c<N)O[c*M+r]=A[r*N+c];}
torch::Tensor transpose(torch::Tensor A){int M=A.size(0),N=A.size(1);auto O=torch::empty({N,M},A.options());
dim3 t(16,16),bl((N+15)/16,(M+15)/16);tr_k<<<bl,t>>>(A.data_ptr<float>(),O.data_ptr<float>(),M,N);return O;}'''

# ------------------------------- Part 3: naive attention -------------------------------
qk_scores = H + r'''
__global__ void qk_k(const float* Q,const float* K,float* S,float sc,int N,int M,int d){
int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<N&&c<M){float s=0;for(int k=0;k<d;k++)s+=Q[r*d+k]*K[c*d+k];S[r*M+c]=s*sc;}}
torch::Tensor qk_scores(torch::Tensor Q,torch::Tensor K,double sc){int N=Q.size(0),d=Q.size(1),M=K.size(0);auto S=torch::empty({N,M},Q.options());
dim3 t(16,16),bl((M+15)/16,(N+15)/16);qk_k<<<bl,t>>>(Q.data_ptr<float>(),K.data_ptr<float>(),S.data_ptr<float>(),(float)sc,N,M,d);return S;}'''

softmax_rows = H + r'''
__global__ void smr_k(const float* S,float* O,int R,int C){int r=blockIdx.x;if(r>=R)return;__shared__ float red[256];
float m=-1e30f;for(int c=threadIdx.x;c<C;c+=blockDim.x)m=fmaxf(m,S[r*C+c]);red[threadIdx.x]=m;__syncthreads();
for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+s]);__syncthreads();}
m=red[0];__syncthreads();float z=0.f;for(int c=threadIdx.x;c<C;c+=blockDim.x){float e=expf(S[r*C+c]-m);O[r*C+c]=e;z+=e;}
red[threadIdx.x]=z;__syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s];__syncthreads();}
float Z=red[0];__syncthreads();for(int c=threadIdx.x;c<C;c+=blockDim.x)O[r*C+c]/=Z;}
torch::Tensor softmax_rows(torch::Tensor S){int R=S.size(0),C=S.size(1);auto O=torch::empty_like(S);
smr_k<<<R,256>>>(S.data_ptr<float>(),O.data_ptr<float>(),R,C);return O;}'''

pv_matmul = H + r'''
__global__ void pv_k(const float* P,const float* V,float* O,int N,int M,int d){
int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<N&&c<d){float s=0;for(int k=0;k<M;k++)s+=P[r*M+k]*V[k*d+c];O[r*d+c]=s;}}
torch::Tensor pv_matmul(torch::Tensor P,torch::Tensor V){int N=P.size(0),M=P.size(1),d=V.size(1);auto O=torch::empty({N,d},P.options());
dim3 t(16,16),bl((d+15)/16,(N+15)/16);pv_k<<<bl,t>>>(P.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),N,M,d);return O;}'''

naive_attention = H + r'''
__global__ void na_k(const float* Q,const float* K,const float* V,float* O,float sc,int N,int M,int d){
int r=blockIdx.x;if(r>=N)return;extern __shared__ float scr[];__shared__ float red[256];
for(int c=threadIdx.x;c<M;c+=blockDim.x){float s=0;for(int k=0;k<d;k++)s+=Q[r*d+k]*K[c*d+k];scr[c]=s*sc;}__syncthreads();
float m=-1e30f;for(int c=threadIdx.x;c<M;c+=blockDim.x)m=fmaxf(m,scr[c]);red[threadIdx.x]=m;__syncthreads();
for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+s]);__syncthreads();}
m=red[0];__syncthreads();float ls=0.f;for(int c=threadIdx.x;c<M;c+=blockDim.x){float e=expf(scr[c]-m);scr[c]=e;ls+=e;}
red[threadIdx.x]=ls;__syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s];__syncthreads();}
float Z=red[0];__syncthreads();for(int j=threadIdx.x;j<d;j+=blockDim.x){float acc=0;for(int c=0;c<M;c++)acc+=scr[c]*V[c*d+j];O[r*d+j]=acc/Z;}}
torch::Tensor naive_attention(torch::Tensor Q,torch::Tensor K,torch::Tensor V,double sc){int N=Q.size(0),d=Q.size(1),M=K.size(0);
auto O=torch::empty({N,d},Q.options());na_k<<<N,256,M*sizeof(float)>>>(Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),(float)sc,N,M,d);return O;}'''

# ------------------------------- Part 4: online softmax math -------------------------------
online_max = H + r'''
__global__ void om_k(const float* m,const float* x,float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=fmaxf(m[i],x[i]);}
torch::Tensor online_max(torch::Tensor m,torch::Tensor x){auto o=torch::empty_like(m);int n=m.numel();int t=256,bl=(n+t-1)/t;
om_k<<<bl,t>>>(m.data_ptr<float>(),x.data_ptr<float>(),o.data_ptr<float>(),n);return o;}'''

correction_factor = H + r'''
__global__ void cf_k(const float* a,const float* b,float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=expf(a[i]-b[i]);}
torch::Tensor correction_factor(torch::Tensor mo,torch::Tensor mn){auto o=torch::empty_like(mo);int n=mo.numel();int t=256,bl=(n+t-1)/t;
cf_k<<<bl,t>>>(mo.data_ptr<float>(),mn.data_ptr<float>(),o.data_ptr<float>(),n);return o;}'''

update_running_sum = H + r'''
__global__ void urs_k(const float* l,const float* a,const float* s,float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=a[i]*l[i]+s[i];}
torch::Tensor update_running_sum(torch::Tensor l,torch::Tensor a,torch::Tensor s){auto o=torch::empty_like(l);int n=l.numel();int t=256,bl=(n+t-1)/t;
urs_k<<<bl,t>>>(l.data_ptr<float>(),a.data_ptr<float>(),s.data_ptr<float>(),o.data_ptr<float>(),n);return o;}'''

rescale_output = H + r'''
__global__ void rso_k(const float* O,const float* a,float* o,int R,int d){int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<R&&c<d)o[r*d+c]=a[r]*O[r*d+c];}
torch::Tensor rescale_output(torch::Tensor O,torch::Tensor a){int R=O.size(0),d=O.size(1);auto o=torch::empty_like(O);
dim3 t(16,16),bl((d+15)/16,(R+15)/16);rso_k<<<bl,t>>>(O.data_ptr<float>(),a.data_ptr<float>(),o.data_ptr<float>(),R,d);return o;}'''

# ------------------------------- Part 5: tiled building blocks -------------------------------
load_tile = H + r'''
__global__ void lt_k(const float* X,float* O,int R,int C){
int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(r<R&&c<C)O[r*C+c]=X[r*C+c];}
torch::Tensor load_tile(torch::Tensor X){int R=X.size(0),C=X.size(1);auto O=torch::empty_like(X);
dim3 t(16,16),bl((C+15)/16,(R+15)/16);lt_k<<<bl,t>>>(X.data_ptr<float>(),O.data_ptr<float>(),R,C);return O;}'''

tile_scores = H + r'''
__global__ void tsc_k(const float* Q,const float* K,float* S,float sc,int N,int M,int d){
int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<N&&c<M){float s=0;for(int k=0;k<d;k++)s+=Q[r*d+k]*K[c*d+k];S[r*M+c]=s*sc;}}
torch::Tensor tile_scores(torch::Tensor Q,torch::Tensor K,double sc){int N=Q.size(0),d=Q.size(1),M=K.size(0);auto S=torch::empty({N,M},Q.options());
dim3 t(16,16),bl((M+15)/16,(N+15)/16);tsc_k<<<bl,t>>>(Q.data_ptr<float>(),K.data_ptr<float>(),S.data_ptr<float>(),(float)sc,N,M,d);return S;}'''

tile_rowmax = H + r'''
__global__ void trm_k(const float* a,float* o,int R,int C){int r=blockIdx.x;if(r>=R)return;float m=-1e30f;
for(int c=threadIdx.x;c<C;c+=blockDim.x)m=fmaxf(m,a[r*C+c]);__shared__ float sm[256];sm[threadIdx.x]=m;__syncthreads();
for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+s]);__syncthreads();}
if(threadIdx.x==0)o[r]=sm[0];}
torch::Tensor tile_rowmax(torch::Tensor a){int R=a.size(0),C=a.size(1);auto o=torch::empty({R},a.options());
trm_k<<<R,256>>>(a.data_ptr<float>(),o.data_ptr<float>(),R,C);return o;}'''

tile_exp = H + r'''
__global__ void tex_k(const float* S,const float* m,float* O,int R,int C){int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<R&&c<C)O[r*C+c]=expf(S[r*C+c]-m[r]);}
torch::Tensor tile_exp(torch::Tensor S,torch::Tensor m){int R=S.size(0),C=S.size(1);auto O=torch::empty_like(S);
dim3 t(16,16),bl((C+15)/16,(R+15)/16);tex_k<<<bl,t>>>(S.data_ptr<float>(),m.data_ptr<float>(),O.data_ptr<float>(),R,C);return O;}'''

tile_rowsum = H + r'''
__global__ void trs_k(const float* a,float* o,int R,int C){int r=blockIdx.x;if(r>=R)return;float v=0.f;
for(int c=threadIdx.x;c<C;c+=blockDim.x)v+=a[r*C+c];__shared__ float sm[256];sm[threadIdx.x]=v;__syncthreads();
for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sm[threadIdx.x]+=sm[threadIdx.x+s];__syncthreads();}
if(threadIdx.x==0)o[r]=sm[0];}
torch::Tensor tile_rowsum(torch::Tensor a){int R=a.size(0),C=a.size(1);auto o=torch::empty({R},a.options());
trs_k<<<R,256>>>(a.data_ptr<float>(),o.data_ptr<float>(),R,C);return o;}'''

accumulate_pv = H + r'''
__global__ void apv_k(const float* P,const float* V,float* O,int N,int M,int d){
int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<N&&c<d){float s=0;for(int k=0;k<M;k++)s+=P[r*M+k]*V[k*d+c];O[r*d+c]=s;}}
torch::Tensor accumulate_pv(torch::Tensor P,torch::Tensor V){int N=P.size(0),M=P.size(1),d=V.size(1);auto O=torch::empty({N,d},P.options());
dim3 t(16,16),bl((d+15)/16,(N+15)/16);apv_k<<<bl,t>>>(P.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),N,M,d);return O;}'''

# ------------------------------- Part 6: fused flash -------------------------------
_FLASH_BODY = r'''
__global__ void %s(const float* Q,const float* K,const float* V,float* O,float sc,int N,int M,int d){
int r=blockIdx.x;if(r>=N)return;if(threadIdx.x!=0)return;
float m=-1e30f,l=0.f,acc[128];for(int j=0;j<d;j++)acc[j]=0.f;
for(int c=0;c<M;c++){float s=0;for(int k=0;k<d;k++)s+=Q[r*d+k]*K[c*d+k];s*=sc;
 float mn=fmaxf(m,s),corr=expf(m-mn),p=expf(s-mn);l=corr*l+p;
 for(int j=0;j<d;j++)acc[j]=corr*acc[j]+p*V[c*d+j];m=mn;}
for(int j=0;j<d;j++)O[r*d+j]=acc[j]/l;}
torch::Tensor %s(torch::Tensor Q,torch::Tensor K,torch::Tensor V,double sc){int N=Q.size(0),d=Q.size(1),M=K.size(0);
auto O=torch::empty({N,d},Q.options());%s<<<N,1>>>(Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),(float)sc,N,M,d);return O;}'''

flash_attention_kernel    = H + (_FLASH_BODY % ("fak_k","flash_attention_kernel","fak_k"))
flash_attention_launcher  = H + (_FLASH_BODY % ("fal_k","flash_attention_launcher","fal_k"))

# ------------------------------- Part 7: causal -------------------------------
causal_mask = H + r'''
__global__ void cmk_k(const float* S,float* O,int N,int M){int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
if(r<N&&c<M)O[r*M+c]=(c>r)?-1e30f:S[r*M+c];}
torch::Tensor causal_mask(torch::Tensor S){int N=S.size(0),M=S.size(1);auto O=torch::empty_like(S);
dim3 t(16,16),bl((M+15)/16,(N+15)/16);cmk_k<<<bl,t>>>(S.data_ptr<float>(),O.data_ptr<float>(),N,M);return O;}'''

flash_attention_causal_kernel = H + r'''
__global__ void fack_k(const float* Q,const float* K,const float* V,float* O,float sc,int N,int d){
int r=blockIdx.x;if(r>=N)return;if(threadIdx.x!=0)return;
float m=-1e30f,l=0.f,acc[128];for(int j=0;j<d;j++)acc[j]=0.f;
for(int c=0;c<=r;c++){float s=0;for(int k=0;k<d;k++)s+=Q[r*d+k]*K[c*d+k];s*=sc;
 float mn=fmaxf(m,s),corr=expf(m-mn),p=expf(s-mn);l=corr*l+p;
 for(int j=0;j<d;j++)acc[j]=corr*acc[j]+p*V[c*d+j];m=mn;}
for(int j=0;j<d;j++)O[r*d+j]=acc[j]/l;}
torch::Tensor flash_attention_causal_kernel(torch::Tensor Q,torch::Tensor K,torch::Tensor V,double sc){int N=Q.size(0),d=Q.size(1);
auto O=torch::empty({N,d},Q.options());fack_k<<<N,1>>>(Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),(float)sc,N,d);return O;}'''

# ================================ oracles + grading ================================
import torch  # noqa: E402
def _r(*s): return torch.randn(*s, device="cuda", dtype=torch.float32)
def _attn(Q,K,V,sc): return torch.softmax(sc*(Q@K.t()),dim=1)@V
def _cattn(Q,K,V,sc):
    N=Q.size(0); S=sc*(Q@K.t()); mask=torch.triu(torch.ones(N,N,device=Q.device,dtype=torch.bool),1)
    return torch.softmax(S.masked_fill(mask,-1e30),dim=1)@V

STEP_SPECS = {
 "vector_add":      dict(decl="torch::Tensor vector_add(torch::Tensor,torch::Tensor);", inp=lambda:(_r(5000),_r(5000)), ora=lambda a,b:a+b),
 "scale_array":     dict(decl="torch::Tensor scale_array(torch::Tensor,double);", inp=lambda:(_r(5000),3.5), ora=lambda a,s:a*s),
 "elementwise_exp": dict(decl="torch::Tensor elementwise_exp(torch::Tensor);", inp=lambda:(_r(5000)*0.5,), ora=lambda a:torch.exp(a)),
 "row_max":         dict(decl="torch::Tensor row_max(torch::Tensor);", inp=lambda:(_r(64,128),), ora=lambda a:a.max(1).values),
 "row_sum":         dict(decl="torch::Tensor row_sum(torch::Tensor);", inp=lambda:(_r(64,128),), ora=lambda a:a.sum(1)),
 "dot_product":     dict(decl="torch::Tensor dot_product(torch::Tensor,torch::Tensor);", inp=lambda:(_r(4096),_r(4096)), ora=lambda a,b:(a*b).sum(), atol=1e-2),
 "matmul":          dict(decl="torch::Tensor matmul(torch::Tensor,torch::Tensor);", inp=lambda:(_r(64,48),_r(48,72)), ora=lambda a,b:a@b, atol=1e-3),
 "transpose":       dict(decl="torch::Tensor transpose(torch::Tensor);", inp=lambda:(_r(64,40),), ora=lambda a:a.t().contiguous()),
 "qk_scores":       dict(decl="torch::Tensor qk_scores(torch::Tensor,torch::Tensor,double);", inp=lambda:(_r(48,32),_r(40,32),0.176), ora=lambda Q,K,s:s*(Q@K.t()), atol=1e-3),
 "softmax_rows":    dict(decl="torch::Tensor softmax_rows(torch::Tensor);", inp=lambda:(_r(48,40),), ora=lambda S:torch.softmax(S,1)),
 "pv_matmul":       dict(decl="torch::Tensor pv_matmul(torch::Tensor,torch::Tensor);", inp=lambda:(torch.softmax(_r(48,40),1),_r(40,32)), ora=lambda P,V:P@V, atol=1e-3),
 "naive_attention": dict(decl="torch::Tensor naive_attention(torch::Tensor,torch::Tensor,torch::Tensor,double);", inp=lambda:(_r(48,32),_r(40,32),_r(40,32),0.176), ora=lambda Q,K,V,s:_attn(Q,K,V,s), atol=1e-3),
 "online_max":      dict(decl="torch::Tensor online_max(torch::Tensor,torch::Tensor);", inp=lambda:(_r(100),_r(100)), ora=lambda m,x:torch.maximum(m,x)),
 "correction_factor":dict(decl="torch::Tensor correction_factor(torch::Tensor,torch::Tensor);", inp=lambda:(_r(100)*0.3,_r(100)*0.3+0.5), ora=lambda a,b:torch.exp(a-b)),
 "update_running_sum":dict(decl="torch::Tensor update_running_sum(torch::Tensor,torch::Tensor,torch::Tensor);", inp=lambda:(_r(100).abs(),_r(100).abs(),_r(100).abs()), ora=lambda l,a,s:a*l+s),
 "rescale_output":  dict(decl="torch::Tensor rescale_output(torch::Tensor,torch::Tensor);", inp=lambda:(_r(48,32),_r(48).abs()), ora=lambda O,a:a.unsqueeze(1)*O),
 "load_tile":       dict(decl="torch::Tensor load_tile(torch::Tensor);", inp=lambda:(_r(32,32),), ora=lambda X:X.clone()),
 "tile_scores":     dict(decl="torch::Tensor tile_scores(torch::Tensor,torch::Tensor,double);", inp=lambda:(_r(32,32),_r(32,32),0.176), ora=lambda Q,K,s:s*(Q@K.t()), atol=1e-3),
 "tile_rowmax":     dict(decl="torch::Tensor tile_rowmax(torch::Tensor);", inp=lambda:(_r(32,32),), ora=lambda a:a.max(1).values),
 "tile_exp":        dict(decl="torch::Tensor tile_exp(torch::Tensor,torch::Tensor);", inp=lambda:(_r(32,32),_r(32)), ora=lambda S,m:torch.exp(S-m.unsqueeze(1))),
 "tile_rowsum":     dict(decl="torch::Tensor tile_rowsum(torch::Tensor);", inp=lambda:(_r(32,32).abs(),), ora=lambda a:a.sum(1)),
 "accumulate_pv":   dict(decl="torch::Tensor accumulate_pv(torch::Tensor,torch::Tensor);", inp=lambda:(_r(32,32),_r(32,32)), ora=lambda P,V:P@V, atol=1e-3),
 "flash_attention_kernel":   dict(decl="torch::Tensor flash_attention_kernel(torch::Tensor,torch::Tensor,torch::Tensor,double);", inp=lambda:(_r(64,64),_r(80,64),_r(80,64),0.125), ora=lambda Q,K,V,s:_attn(Q,K,V,s), atol=2e-3),
 "flash_attention_launcher": dict(decl="torch::Tensor flash_attention_launcher(torch::Tensor,torch::Tensor,torch::Tensor,double);", inp=lambda:(_r(64,64),_r(80,64),_r(80,64),0.125), ora=lambda Q,K,V,s:_attn(Q,K,V,s), atol=2e-3),
 "causal_mask":     dict(decl="torch::Tensor causal_mask(torch::Tensor);", inp=lambda:(_r(40,40),), ora=lambda S:torch.where(torch.arange(40,device=S.device)[None,:]>torch.arange(40,device=S.device)[:,None],torch.full_like(S,-1e30),S)),
 "flash_attention_causal_kernel": dict(decl="torch::Tensor flash_attention_causal_kernel(torch::Tensor,torch::Tensor,torch::Tensor,double);", inp=lambda:(_r(64,64),_r(64,64),_r(64,64),0.125), ora=lambda Q,K,V,s:_cattn(Q,K,V,s), atol=2e-3),
}


def grade(ns, name):
    """Compile ns[name] (CUDA source) and check mod.<name> against the torch oracle.
    No-ops without a GPU so CI/verify pass on CPU-only machines."""
    if not have_gpu():
        return
    spec = STEP_SPECS[name]
    mod = compile_cuda(ns[name], [name], cpp_decls=spec["decl"], tag=name)
    inp = spec["inp"]()
    out = getattr(mod, name)(*inp)
    exp = spec["ora"](*inp)
    atol = spec.get("atol", 1e-4); rtol = spec.get("rtol", 1e-3)
    assert tuple(out.shape) == tuple(exp.shape), f"{name}: shape {tuple(out.shape)} != {tuple(exp.shape)}"
    assert torch.allclose(out, exp, atol=atol, rtol=rtol), f"{name}: max abs err {(out-exp).abs().max().item():.2e}"
