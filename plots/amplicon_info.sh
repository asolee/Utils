#!/bin/bash

#read arguments
bed_file=''
bam_file=''
genome=''

print_usage() {
  printf "Usage: ..."
}

while getopts 'a:b:g:' flag; do
  case "${flag}" in
    a) bed_file="${OPTARG}" ;;
    b) bam_file="${OPTARG}" ;;
    g) genome="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

#annotate bed file
out_bed=$(basename ${bed_file%.*})_annotate.bed
bed_annotation $bed_file -g $genome -o $out_bed --extended

#generate samtools stats
out_stats=$(basename ${bam_file%.*})_cov_stats_tsv
samtools bedcov $out_bed $bam_file > $out_stats
