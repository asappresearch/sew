kenlm_build_bin=`realpath $1`
lm_dir=save-lm/kenlm/librispeech-official
mkdir -p $lm_dir
cd $lm_dir
echo "downloading"
if ! [ -f lexicon.lst ]; then
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
    mv librispeech_lexicon.lst lexicon.lst
fi

if ! [ -f 4-gram.arpa ]; then
    wget https://www.openslr.org/resources/11/4-gram.arpa.gz
    gzip -d 4-gram.arpa.gz
fi

echo "processing"
${kenlm_build_bin}/build_binary 4-gram.arpa 4gram.bin
