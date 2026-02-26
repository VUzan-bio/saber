# Reference Data

Download these files before running SABER:

## M. tuberculosis H37Rv

```bash
# Reference genome
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.fna.gz
gunzip GCF_000195955.2_ASM19595v2_genomic.fna.gz
mv GCF_000195955.2_ASM19595v2_genomic.fna H37Rv.fasta

# Annotation
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.gff.gz
gunzip GCF_000195955.2_ASM19595v2_genomic.gff.gz
mv GCF_000195955.2_ASM19595v2_genomic.gff H37Rv.gff3

# Bowtie2 index
bowtie2-build H37Rv.fasta H37Rv
```

## Human reference (optional, for blood-based specificity)

```bash
# GRCh38 — only needed if screening off-targets against human genome
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz
bowtie2-build GRCh38.fna GRCh38
```
