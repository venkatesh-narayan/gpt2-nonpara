import transformers
import torch
from lm_eval.base import BaseLM
import os

class knnlmHFLM(BaseLM):
    def __init__(self, dstore_mmap, faiss_index, num_shards, lmbda=0.25, stride=512, save_knnlm_dstore=False, knnlm=True,
                 device='cuda:0', pretrained='gpt2-large', revision='main', subfolder=None, tokenizer=None, batch_size=1,
                 root_path='/projects/tir3/users/junxianh/venkaten/gpt2-nonpara'):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.root_path = root_path
        config = transformers.AutoConfig.from_pretrained(pretrained,
                                                         cache_dir=None,
                                                         revision=revision + ("/" + subfolder if subfolder is not None else ""),
                                                         use_auth_token=None)
        config.stride = stride
        config.save_knnlm_dstore = save_knnlm_dstore
        config.knnlm = knnlm
        config.dstore_mmap = os.path.join(root_path, dstore_mmap)
        config.dstore_size = self.get_dstore_size(num_shards)
        config.faiss_index = os.path.join(root_path, faiss_index)
        config.lmbda = lmbda

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.knnlmGPT2LMHeadModel.from_pretrained(pretrained,
                                                                      revision=revision + ("/" + subfolder if subfolder is not None else ""),
                                                                      config=config).to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained if tokenizer is None else tokenizer,
                                                                    revision=revision,
                                                                    subfolder=subfolder,
                                                                    use_fast=True)

        assert isinstance(self.tokenizer, (
            transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
            transformers.T5Tokenizer, transformers.T5TokenizerFast,
        )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
            assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373], \
                self.tokenizer.encode('hello\n\nhello')

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    def get_dstore_size(self, num_shards):
        dstore_size = 0

        f = open(os.path.join(self.root_path, 'dstore_sizes.txt'), 'r')
        for i, line in enumerate(f):
            if i == num_shards:
                break

            dstore_size += int(line.split(' ')[1])

        f.close()
        return dstore_size

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt2.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )


# for backwards compatibility
knnlmGPT2LM = knnlmHFLM
