from tasks.tts.speech_base import SpeechBaseTask
import os
import traceback
import numpy as np
import torch
import torch.optim
import torch.utils.data
from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.commons.tensor_utils import tensors_to_scalars
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

def plot_attn(p_attn, base_fn, gen_dir):
    if len(base_fn.split('/')) == 2:
        spk, fname = base_fn.split('/')
    else:
        _, _, spk, _, fname = base_fn.split('/')
    new_name = f"{gen_dir}/attn/{spk}/{fname}"
    os.makedirs(os.path.dirname(new_name), exist_ok=True)
    attn_np = p_attn.squeeze(0).detach().cpu().numpy()

    n_heads = attn_np.shape[0]
    for head in range(n_heads):
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_np[head],
                    cmap='viridis',
                    xticklabels=5,
                    yticklabels=5)
        plt.title(f'Attention Map for Head {head}')
        plt.xlabel('Key Sequence Index')
        plt.ylabel('Query Sequence Index')
        plt.savefig(f"{new_name}_h{head}.png", bbox_inches='tight', dpi=300)
        plt.close()

class FillerSpeechTask(SpeechBaseTask):
    def __init__(self):
        super(FillerSpeechTask, self).__init__()
        self.data_stats_mean = torch.tensor(self.hparams.data_ms_stats['mean'])
        self.data_stats_std  = torch.tensor(self.hparams.data_ms_stats['std'])

    def normalize(self, data, mu, std):
        if not isinstance(mu, (float, int)):
            if isinstance(mu, list):
                mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
            elif isinstance(mu, torch.Tensor):
                mu = mu.to(data.device)
            elif isinstance(mu, np.ndarray):
                mu = torch.from_numpy(mu).to(data.device)
            mu = mu.unsqueeze(-1)

        if not isinstance(std, (float, int)):
            if isinstance(std, list):
                std = torch.tensor(std, dtype=data.dtype, device=data.device)
            elif isinstance(std, torch.Tensor):
                std = std.to(data.device)
            elif isinstance(std, np.ndarray):
                std = torch.from_numpy(std).to(data.device)
            std = std.unsqueeze(-1)

        return (data - mu) / std

    def denormalize(self, data, mu, std):
        if not isinstance(mu, float):
            if isinstance(mu, list):
                mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
            elif isinstance(mu, torch.Tensor):
                mu = mu.to(data.device)
            elif isinstance(mu, np.ndarray):
                mu = torch.from_numpy(mu).to(data.device)
            mu = mu.unsqueeze(-1)

        if not isinstance(std, float):
            if isinstance(std, list):
                std = torch.tensor(std, dtype=data.dtype, device=data.device)
            elif isinstance(std, torch.Tensor):
                std = std.to(data.device)
            elif isinstance(std, np.ndarray):
                std = torch.from_numpy(std).to(data.device)
            std = std.unsqueeze(-1)

        return data * std + mu

    def forward(self, sample, infer=False, *args, **kwargs):
        x = sample["txt_tokens"]  # [B, T_t]
        x_lengths = sample["txt_lengths"]
        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]
        f_conds = sample['f_conds'] # [B, T_t]
        dur_conds = sample["dur_conds"] # [B, T_t]
        wf = sample['wfs'] # [B, T_w]
        wf_lengths = sample['wf_lengths']
        spk = sample["mels"].clone().detach().transpose(1,2)
        y = self.normalize(y, self.data_stats_mean, self.data_stats_std)

        if not infer:
            f0 = sample['f0']
            uv = sample['uv']
            mel2ph = sample['mel2ph']

            output, pitch_temp = self.model.compute_loss(
                x,
                x_lengths,
                y.transpose(1, 2),
                y_lengths,
                f0=f0,
                uv=uv,
                mel2ph=mel2ph,
                spk=spk,
                spk_lengths=y_lengths,
                f_conds=f_conds,
                dur_conds=dur_conds,
                wf=wf,
                wf_lengths=wf_lengths,
                out_size=self.hparams["out_size"],
            )
            losses = {}
            losses["dur_loss"] = output["dur_loss"]
            losses["prior_loss"] = output["prior_loss"]
            losses["diff_loss"] = output["diff_loss"]

            self.add_pitch_loss(pitch_temp, sample, losses)

            return losses, output
        else:
            output = self.model(x, x_lengths, spk=spk, spk_lengths=y_lengths, f_conds=f_conds, dur_conds=dur_conds,
                                wf=wf, wf_lengths=wf_lengths, n_timesteps=int(self.hparams['n_timesteps']))
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], _ = self(sample)
        outputs["nsamples"] = sample["nsamples"]


        if (
            self.global_step % self.hparams["valid_infer_interval"] == 0
            and batch_idx < self.hparams["num_valid_plots"]
        ):
            model_out = self(sample, infer=True)
            self.save_valid_result(sample, batch_idx, model_out)

        outputs = tensors_to_scalars(outputs)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = self.hparams["audio_sample_rate"]
        gt = sample["mels"]
        pred = self.denormalize(model_out["mel_out"], self.data_stats_mean, self.data_stats_std)
        prior = self.denormalize(model_out["encoder_outputs"], self.data_stats_mean, self.data_stats_std)
        attn = model_out["attn"].cpu().numpy()

        self.plot_mel(batch_idx, [gt[0], prior[0], pred[0]], title=f"mel_{batch_idx}")
        self.logger.add_image(
            f"plot_attn_{batch_idx}", self.plot_alignment(attn[0]), self.global_step
        )

        wav_pred = self.vocoder.spec2wav(pred[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}", wav_pred, self.global_step, sr)

        wav_pred = self.vocoder.spec2wav(prior[0].cpu())
        self.logger.add_audio(f"wav_prior_{batch_idx}", wav_pred, self.global_step, sr)

        if self.global_step <= self.hparams["valid_infer_interval"]:
            wav_gt = self.vocoder.spec2wav(gt[0].cpu())
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)

    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert (
            sample["txt_tokens"].shape[0] == 1
        ), "only support batch_size=1 in inference"
        outputs = self(sample, infer=True)

        text = sample["text"][0]
        item_name = sample["item_name"][0]
        tokens = sample["txt_tokens"][0].cpu().numpy()
        mel_gt = sample["mels"][0].cpu().numpy()
        mel_pred = self.denormalize(outputs["mel_out"][0], self.data_stats_mean, self.data_stats_std).cpu().numpy()
        mel2ph_item = sample.get("mel2ph")
        if mel2ph_item is not None:
            mel2ph = mel2ph_item[0].cpu().numpy()
        else:
            mel2ph = None
        mel2ph_pred_item = outputs.get("mel2ph")
        if mel2ph_pred_item is not None:
            mel2ph_pred = mel2ph_pred_item[0].cpu().numpy()
        else:
            mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)

        base_fn = item_name
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)

        audio_sample_rate = self.hparams["audio_sample_rate"]
        out_wav_norm = self.hparams["out_wav_norm"]
        mel_vmin = self.hparams["mel_vmin"]
        mel_vmax = self.hparams["mel_vmax"]
        save_mel_npy = self.hparams["save_mel_npy"]


        self.saving_result_pool.add_job(
            self.save_result,
            args=[
                wav_pred,
                mel_pred,
                base_fn,
                gen_dir,
                str_phs,
                mel2ph_pred,
                None,
                audio_sample_rate,
                out_wav_norm,
                mel_vmin,
                mel_vmax,
                save_mel_npy,
            ],
        )
        if self.hparams["save_gt"]:
            gt_name = base_fn + "_gt"
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[
                    wav_gt,
                    mel_gt,
                    gt_name,
                    gen_dir,
                    str_phs,
                    mel2ph,
                    None,
                    audio_sample_rate,
                    out_wav_norm,
                    mel_vmin,
                    mel_vmax,
                    save_mel_npy,
                ],
            )

        return {
            "item_name": item_name,
            "text": text,
            "ph_tokens": self.token_encoder.decode(tokens.tolist()),
            "wav_fn_pred": base_fn,
            "wav_fn_gt": base_fn + "_gt",
        }


    @staticmethod
    def save_result(
        wav_out,
        mel,
        base_fn,
        gen_dir,
        str_phs=None,
        mel2ph=None,
        alignment=None,
        audio_sample_rate=16000,
        out_wav_norm=True,
        mel_vmin=-6,
        mel_vmax=1.5,
        save_mel_npy=False,
    ):
        if len(base_fn.split('/')) == 2:
            spk, fname = base_fn.split('/')
        else:
            _, _, spk, _, fname = base_fn.split('/')
        new_name = f"{gen_dir}/wavs/{spk}/{fname}.wav"
        os.makedirs(os.path.dirname(new_name), exist_ok=True)


        save_wav(
            wav_out,
            new_name,
            audio_sample_rate,
            norm=out_wav_norm,
        )
        # @@@@@@@@@@@@@@@@@@@@@@
        PLOT_FIGURE=False
        if PLOT_FIGURE:
            fig = plt.figure(figsize=(14, 10))
            spec_vmin = mel_vmin
            spec_vmax = mel_vmax
            heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
            fig.colorbar(heatmap)
            try:
                f0 = extract_pitch_simple(wav_out)
                f0 = f0 / 10 * (f0 > 0)
                plt.plot(f0, c="white", linewidth=1, alpha=0.6)
                if mel2ph is not None and str_phs is not None:
                    decoded_txt = str_phs.split(" ")
                    dur = mel2token_to_dur(
                        torch.LongTensor(mel2ph)[None, :], len(decoded_txt)
                    )[0].numpy()
                    dur = [0] + list(np.cumsum(dur))
                    for i in range(len(dur) - 1):
                        shift = (i % 20) + 1
                        plt.text(dur[i], shift, decoded_txt[i])
                        plt.hlines(
                            shift,
                            dur[i],
                            dur[i + 1],
                            colors="b" if decoded_txt[i] != "|" else "black",
                        )
                        plt.vlines(
                            dur[i],
                            0,
                            5,
                            colors="b" if decoded_txt[i] != "|" else "black",
                            alpha=1,
                            linewidth=1,
                        )
                plt.tight_layout()
                os.makedirs(os.path.dirname(f"{gen_dir}/plot/{base_fn}.png"), exist_ok=True)
                plt.savefig(f"{gen_dir}/plot/{base_fn}.png", format="png")
                plt.close(fig)
                if save_mel_npy:
                    np.save(f"{gen_dir}/mel_npy/{base_fn}", mel)
                if alignment is not None:
                    fig, ax = plt.subplots(figsize=(12, 16))
                    im = ax.imshow(
                        alignment, aspect="auto", origin="lower", interpolation="none"
                    )
                    decoded_txt = str_phs.split(" ")
                    ax.set_yticks(np.arange(len(decoded_txt)))
                    ax.set_yticklabels(list(decoded_txt), fontsize=6)
                    fig.colorbar(im, ax=ax)
                    fig.savefig(f"{gen_dir}/attn_plot/{base_fn}_attn.png", format="png")
                    plt.close(fig)
            except Exception:
                traceback.print_exc()
        return None


    def add_pitch_loss(self, output, sample, losses):
        cwt_spec = sample["cwt_spec"]
        f0_mean = sample["f0_mean"]
        uv = sample["uv"]
        mel2ph = sample["mel2ph"]
        f0_std = sample["f0_std"]
        cwt_pred = output["cwt"][:, :, :10]
        f0_mean_pred = output["f0_mean"]
        f0_std_pred = output["f0_std"]
        nonpadding = (mel2ph != 0).float()
        losses["f0_cwt"] = F.l1_loss(cwt_pred, cwt_spec) * self.hparams["lambda_f0"]

        assert output["cwt"].shape[-1] == 11
        uv_pred = output["cwt"][:, :, -1]
        losses["uv"] = (
            (
                F.binary_cross_entropy_with_logits(uv_pred, uv, reduction="none")
                * nonpadding
            ).sum()
            / nonpadding.sum()
            * self.hparams["lambda_uv"]
        )
        losses["f0_mean"] = F.l1_loss(f0_mean_pred, f0_mean) * self.hparams["lambda_f0"]
        losses["f0_std"] = F.l1_loss(f0_std_pred, f0_std) * self.hparams["lambda_f0"]