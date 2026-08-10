/* Biais Attention/FFN — valeurs réelles (float)
 * Quantification Q8.8 effectuée automatiquement par le constructeur ap_fixed
 * à la compilation HLS — pas de division au runtime, coût matériel nul.
 * NE PAS multiplier par 256 ici : contrairement aux poids (chargés via AXI,
 * bits bruts réinterprétés), ces constantes sont compilées directement en
 * HLS et doivent rester en valeur réelle.
 * Test Accuracy: 100.00 % */

#ifndef ATTENTION_BIAS_PARAMS_H
#define ATTENTION_BIAS_PARAMS_H

static const data_t in_proj_bias_q[NUM_HEADS][D_HEAD] = {
    { 0.01249486f, 0.15538631f, 0.04766571f, 0.01613167f, -0.14171529f, 0.09343209f, -0.01702736f, -0.06671014f },
    { 0.09783374f, 0.12543648f, 0.15801814f, 0.05470140f, -0.09736940f, -0.13800149f, 0.13543338f, 0.09782124f }
};
static const data_t in_proj_bias_k[NUM_HEADS][D_HEAD] = {
    { -0.00000044f, 0.00000349f, -0.00000136f, -0.00000101f, 0.00000047f, -0.00000098f, -0.00000121f, 0.00000139f },
    { -0.00000528f, 0.00000329f, -0.00000085f, -0.00000285f, 0.00000250f, 0.00000544f, 0.00000080f, -0.00000022f }
};
static const data_t in_proj_bias_v[NUM_HEADS][D_HEAD] = {
    { 0.00237485f, 0.06459333f, 0.02052898f, 0.05476446f, 0.00852073f, -0.00050630f, -0.05221909f, 0.04464423f },
    { 0.01836387f, 0.00189625f, 0.00442327f, 0.07021476f, 0.02896013f, 0.02917624f, 0.04081810f, 0.01141981f }
};
static const data_t out_proj_bias_q[D_MODEL] = { -0.01111767f, -0.00213841f, -0.04043913f, -0.01444229f, -0.01557749f, 0.00204321f, -0.01735768f, 0.05526396f, 0.05034024f, -0.03302771f, -0.04256776f, 0.07107101f, -0.02045363f, -0.01431303f, 0.05494328f, -0.03755838f };
static const data_t ffn0_bias_q[D_FF] = { -0.04208947f, 0.13222882f, 0.10590526f, -0.09459440f, 0.22900051f, -0.01329507f, -0.02860511f, 0.17172424f, -0.04338197f, -0.08641671f, -0.07573420f, -0.09928919f, -0.11779384f, 0.06062491f, 0.20024651f, -0.01780170f, 0.02609641f, 0.04796306f, 0.15043186f, -0.12724704f, -0.17082204f, 0.18452628f, 0.22647588f, 0.05083919f, -0.02621777f, 0.16345401f, 0.09322930f, 0.11132289f, 0.21088001f, 0.09774239f, 0.10811301f, -0.01777509f };
static const data_t ffn2_bias_q[D_MODEL] = { -0.16110682f, -0.10831407f, -0.05125262f, 0.13376917f, -0.15883808f, -0.12453017f, -0.05516414f, 0.11863425f, 0.10534330f, 0.13824409f, 0.13030754f, -0.09164829f, -0.03989961f, -0.14719337f, 0.02319358f, 0.05922856f };

#endif /* ATTENTION_BIAS_PARAMS_H */
