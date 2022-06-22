import transformers
import torch
from lm_eval.base import BaseLM
import os

class knnlmHFLM(BaseLM):
    def __init__(self, dstore_mmap, faiss_index, dstore_sizes_path, num_shards, lmbda=0.25, stride=512, device='cuda:0',
                 pretrained='gpt2-large', use_gpu_faiss=True, exclude_shards='[]', include_shards='[]', revision='main',
                 subfolder=None, tokenizer=None, batch_size=1, root_path='/projects/tir3/users/junxianh/venkaten/gpt2-nonpara'):
        super().__init__()

        # if passing in arguments, they'll probably be stored as strings -- convert to correct dtype
        batch_size = int(batch_size)
        stride = int(stride)
        lmbda = float(lmbda)
        num_shards = int(num_shards)

        # for now, include_shards and exclude_shards aren't being used, so these next set of lines don't matter too much
        exclude_shards = exclude_shards[1:-1] # remove brackets
        include_shards = include_shards[1:-1]

        exclude_shards = exclude_shards.split(' ') # assuming that they're space separated bc the model args are comma separated
        include_shards = include_shards.split(' ')

        if exclude_shards == ['']:
            exclude_shards = []
        else:
            exclude_shards = [int(shard) for shard in exclude_shards] # convert to int

        if include_shards == ['']:
            include_shards = []
        else:
            include_shards = [int(shard) for shard in include_shards]

        shard_idxs_used = [x for x in range(num_shards) if x not in exclude_shards] # remove parts from exclude_shards
        shard_idxs_used.extend(include_shards) # add all the include shards

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
        config.save_knnlm_dstore = False
        config.knnlm = True # always do evaluation here
        config.dstore_mmap = os.path.join(root_path, dstore_mmap)
        config.dstore_size = self.get_dstore_size(dstore_sizes_path, num_shards)
        config.faiss_index = os.path.join(root_path, faiss_index)
        config.lmbda = lmbda
        config.num_shards = num_shards
        config.use_gpu_faiss = use_gpu_faiss
        config.shard_idxs_used = shard_idxs_used

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        # using GPT2Tokenizer instead of AutoTokenizer
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(pretrained if tokenizer is None else tokenizer,
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

        config.tokenizer = self.tokenizer

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.knnlmGPT2LMHeadModel.from_pretrained(pretrained,
                                                                      revision=revision + ("/" + subfolder if subfolder is not None else ""),
                                                                      config=config).to(self.device)

        self.gpt2.eval()


        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    def get_dstore_size(self, dstore_sizes_path, num_shards):
        dstore_size = 0

        f = open(os.path.join(self.root_path, dstore_sizes_path), 'r')
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
