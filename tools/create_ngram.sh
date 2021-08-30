kenlm_build_bin=$1
input=$2
output=$3
n=$4

mkdir -p $output

arpa=${output}/${n}gram.arpa
bin=${output}/${n}gram.bin


cat $input | tr '[a-z]' '[A-Z'] | ${kenlm}/lmplz --skip_symbols -o ${n} > $arpa
${kenlm_build_bin}/build_binary $arpa $bin

echo 'create lexicon'
python felix_scripts/create_lexicon.py $input ${output}/lexicon.lst
rm -f $arpa