/* Provides the source code listings given in the paper:

    https://arxiv.org/abs/2410.18026

    "EON: A practical energy-preserving rough diffuse BRDF",
        Jamie Portsmouth, Peter Kutz, Stephen Hill

  Note that this implementation assumes throughout that the directions are
  specified in a local space where the $z$-direction aligns with the surface normal.
*/

const float PI            = 3.14159265f;
const float constant1_FON = 0.5f - 2.0f / (3.0f * PI);
const float constant2_FON = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// FON directional albedo (analytic)
float E_FON_exact(float mu, float r)
{
    #define safe_acos(x) (acos(clamp(x, -1.0, 1.0)))
    float AF = 1.0f / (1.0f + constant1_FON * r); // FON A coeff.
    float BF = r * AF;                            // FON B coeff.
    float Si = sqrt(1.0f - (mu * mu));
    float G = Si * (safe_acos(mu) - Si * mu)
            + (2.0f / 3.0f) * ((Si / mu) * (1.0f - (Si * Si * Si)) - Si);
    return AF + (BF/PI) * G;
}

// FON directional albedo (approx.)
 float E_FON_approx(float mu, float r)
  {
      float mucomp = 1.0f - mu;
      float mucomp2 = mucomp * mucomp;
      const mat2 Gcoeffs = mat2(0.0571085289f, -0.332181442f,
                                0.491881867f, 0.0714429953f);
      float GoverPi = dot(Gcoeffs * vec2(mucomp, mucomp2), vec2(1.0f, mucomp2));
      return (1.0f + r * GoverPi) / (1.0f + constant1_FON * r);
  }

////////////////////////////////////////////////////////////////////////////////
// EON BRDF
////////////////////////////////////////////////////////////////////////////////

  // Evaluates EON BRDF value, given inputs:
  //       rho = single-scattering albedo parameter
  //         r = roughness in [0, 1]
  //  wi_local = direction of incident ray (directed away from vertex)
  //  wo_local = direction of outgoing ray (directed away from vertex)
  //     exact = flag to select exact or fast approx. version
  //
  vec3 f_EON(vec3 rho, float r, vec3 wi_local, vec3 wo_local, bool exact)
  {
      float mu_i = wi_local.z;                               // input angle cos
      float mu_o = wo_local.z;                               // output angle cos
      float s = dot(wi_local, wo_local) - mu_i * mu_o;       // QON $s$ term
      float sovertF = s > 0.0f ? s / max(mu_i, mu_o) : s;    // FON $s/t$
      float AF = 1.0f / (1.0f + constant1_FON * r);          // FON $A$ coeff.
      vec3 f_ss = (rho/PI) * AF * (1.0f + r * sovertF); // single-scatter
      float EFo = exact ? E_FON_exact(mu_o, r):              // FON $w_o$ albedo (exact)
                          E_FON_approx(mu_o, r);             // FON $w_o$ albedo (approx)
      float EFi = exact ? E_FON_exact(mu_i, r):              // FON $w_i$ albedo (exact)
                          E_FON_approx(mu_i, r);             // FON $w_i$ albedo (approx)
      float avgEF = AF * (1.0f + constant2_FON * r);  // avg. albedo
      vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0f) - rho * (1.0f - avgEF));
      const float eps = 1.0e-7f;
      vec3 f_ms = (rho_ms/PI) * max(eps, 1.0f - EFo) // multi-scatter lobe
                              * max(eps, 1.0f - EFi)
                              / max(eps, 1.0f - avgEF);
      return f_ss + f_ms;
  }

////////////////////////////////////////////////////////////////////////////////
// EON directional albedo
////////////////////////////////////////////////////////////////////////////////

  vec3 E_EON(vec3 rho, float r, vec3 wi_local, bool exact)
  {
      float mu_i = wi_local.z;                       // input angle cos
      float AF = 1.0f / (1.0f + constant1_FON * r);  // FON $A$ coeff.
      float EF = exact ? E_FON_exact(mu_i, r):       // FON $w_i$ albedo (exact)
                         E_FON_approx(mu_i, r);      // FON $w_i$ albedo (approx)
      float avgEF = AF * (1.0f + constant2_FON * r); // average albedo
      vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0f) - rho * (1.0f - avgEF));
      return rho * EF + rho_ms * (1.0f - EF);
  }

////////////////////////////////////////////////////////////////////////////////
// EON importance sampling
////////////////////////////////////////////////////////////////////////////////

void ltc_terms(float mu, float r,
               out float a, out float b, out float c, out float d)
{
    a = 1.0 + r*(0.303392 + (-0.518982 + 0.111709*mu)*mu + (-0.276266 + 0.335918*mu)*r);
    b = r*(-1.16407 + 1.15859*mu + (0.150815 - 0.150105*mu)*r)/(mu*mu*mu - 1.43545);
    c = 1.0 + (0.20013 + (-0.506373 + 0.261777*mu)*mu)*r;
    d = ((0.540852 + (-1.01625 + 0.475392*mu)*mu)*r)/(-1.0743 + mu*(0.0725628 + mu));
}

// (NB, uses opposite convention that wo_local is the incident/camera ray direction)
vec4 cltc_sample(in vec3 wo_local, float r, float u1, float u2)
{
    float a, b, c, d; ltc_terms(wo_local.z, r, a, b, c, d);  // coeffs of LTC $M$
    float R = sqrt(u1); float phi = 2.0 * PI * u2;           // CLTC sampling
    float x = R * cos(phi); float y = R * sin(phi);          // CLTC sampling
    float vz = 1.0 / sqrt(d*d + 1.0);                        // CLTC sampling factors
    float s = 0.5 * (1.0 + vz);                              // CLTC sampling factors
    x = -mix(sqrt(1.0 - y*y), x, s);                         // CLTC sampling
    vec3 wh = vec3(x, y, sqrt(max(1.0 - (x*x + y*y), 0.0))); // $w_h$ sample via CLTC
    float pdf_wh = wh.z / (PI * s);                          // PDF of $w_h$ sample
    vec3 wi = vec3(a*wh.x + b*wh.z, c*wh.y, d*wh.x + wh.z);  // $M w_h$ (unnormalized)
    float len = length(wi);                                  // $|M w_h| = 1/|M^{-1} w_h|$
    float detM = c*(a - b*d);                                // $|M|$
    float pdf_wi = pdf_wh * len*len*len / detM;              // $w_i$ sample PDF
    mat3 fromLTC = orthonormal_basis_ltc(wo_local);          // transform $w_i$ to world space
    wi = normalize(fromLTC * wi);                            // transform $w_i$ to world space
    return vec4(wi, pdf_wi);
}

// (NB, uses opposite convention that wo_local is the incident/camera ray direction)
float cltc_pdf(in vec3 wo_local, in vec3 wi_local, float r)
{
    mat3 toLTC = transpose(orthonormal_basis_ltc(wo_local));                 // transform $w_i$ to LTC space
    vec3 wi = toLTC * wi_local;                                               // transform $w_i$ to LTC space
    float a, b, c, d; ltc_terms(wo_local.z, r, a, b, c, d);                  // coeffs of LTC $M$
    float detM = c*(a - b*d);                                                // $|M|$
    vec3 wh = vec3(c*(wi.x - b*wi.z), (a - b*d)*wi.y, -c*(d*wi.x - a*wi.z)); // $\mathrm{adj}(M) wi$
    float lensq = dot(wh, wh);                                               // $|M| |M^{-1} wi|$
    float vz = 1.0 / sqrt(d*d + 1.0);                                        // CLTC sampling factors
    float s = 0.5 * (1.0 + vz);                                              // CLTC sampling factors
    float pdf = sqr(detM / lensq) * max(wh.z, 0.0) / (PI * s);               // $w_i$ sample PDF
    return pdf;
}

vec3 uniform_lobe_sample(float u1, float u2)
{
    float z = u1;
    float R = sqrt(1.0 - z*z); float phi = 2.0 * PI * u2;
    float x = R * cos(phi); float y = R * sin(phi);
    return vec3(x, y, z);
}

vec4 sample_EON(in vec3 wo_local, float r, float u1, float u2)
{
    float mu = wo_local.z;
    float P_u = pow(r, 0.1) * (0.162925 + mu*(-0.372058 + (0.538233 - 0.290822*mu)*mu));
    float P_c = 1.0 - P_u;
    vec4 wi; float pdf_C;
    if (u1 <= P_u) {
        u1 = u1 / P_u;
        wi.xyz = uniform_lobe_sample(u1, u2);
        pdf_C = cltc_pdf(wo_local, wi.xyz, r); }
    else {
        u1 = (u1 - P_u) / P_c;
        wi = cltc_sample(wo_local, r, u1, u2);
        pdf_C = wi.w; }
    const float pdf_U = 1.0 / (2.0 * PI);
    wi.w = P_u*pdf_U + P_c*pdf_C;
    return wi;
}

float pdf_EON(in vec3 wo_local, in vec3 wi_local, float r)
{
    float mu = wo_local.z;
    float P_u = pow(r, 0.1) * (0.162925 + mu*(-0.372058 + (0.538233 - 0.290822*mu)*mu));
    float P_c = 1.0 - P_u;
    float pdf_cltc = cltc_pdf(wo_local, wi_local, r);
    const float pdf_U = 1.0 / (2.0 * PI);
    return P_u*pdf_U + P_c*pdf_cltc;
}


////////////////////////////////////////////////////////////////////////////////
// Example usage in typical evaluate/sample/albedo routines for a renderer.
////////////////////////////////////////////////////////////////////////////////

/*
    Note that we assume the usual convention for unidirectional path tracing,
    where the direction of the outgoing ray $\omega_o$ is known, and the
    incident ray direction $\omega_i$ is the one being sampled.

    Both the incident ray direction $\omega_i$ and the outgoing ray direction $\omega_o$
    are oriented to point away from the surface, i.e., the incident ray is in the opposite direction
    to incident photons, while the outgoing ray is parallel to the outgoing photons.

    Also note that the sampling would work equally well for the reverse case,
    when tracing paths in the direction of light flow.
*/

uniform vec3 albedo;
uniform float roughness;

const float DENOM_TOLERANCE = 1.0e-7;

vec3 diffuse_brdf_evaluate(in vec3 wi_local, in vec3 wo_local,
                           inout float pdf_wo_local)
{
    if (wi_local.z < DENOM_TOLERANCE || wo_local.z < DENOM_TOLERANCE) return vec3(0.0);
    pdf_wo_local = pdf_EON(wi_local, wo_local, roughness);
    return f_EON(albedo, roughness, wi_local, wo_local, true);
}

vec3 diffuse_brdf_sample(in vec3 wo_local, inout uint rndSeed,
                         out vec3 wi_local, out float pdf_wi_local)
{
    if (wo_local.z < DENOM_TOLERANCE) return vec3(0.0);
    float u1 = rand(rndSeed); float u2 = rand(rndSeed);
    vec4 wiP = sample_EON(wo_local, roughness, u1, u2);
    wi_local     = wiP.xyz;
    pdf_wi_local = wiP.w;
    return f_EON(albedo, roughness, wi_local, wo_local, true);
}

vec3 diffuse_brdf_albedo(in vec3 wo_local, inout uint rndSeed)
{
    if (wo_local.z < DENOM_TOLERANCE) return vec3(0.0);
    return E_EON(albedo, roughness, wo_local);
}