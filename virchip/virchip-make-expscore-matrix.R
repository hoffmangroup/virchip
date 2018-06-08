options(stringsAsFactors=FALSE)
args = commandArgs(trailingOnly=TRUE)
if(length(args) != 6){
    cat("Usage: Rscript: chip_rna_cor.R <RnaPath> <ChipMatPath> <OutPath> <Window> <NumGenes> <Chrom>\n")
    cat("Please provide all required 6 positional arguments")
}
cat("Proceeding with 5 provided arguments\n")
print(args)
# Arguments
rna_path = args[1]
chip_path = args[2]
out_path = args[3]

# Constants
window = as.numeric(args[4])
num_genes = as.numeric(args[5])
chrom = args[6]




get_rna_df = function(rna_path, cells, num_genes=2000){
    rna_mat = read.csv(rna_path, sep="\t", row.names=1, header=T, check.names=FALSE)
    cells = cells[cells %in% colnames(rna_mat)]
    name_genes = rownames(rna_mat)
    rna_mat = rna_mat[, cells]
    rna_mat = rna_mat[order(apply(rna_mat, 1, var), decreasing=TRUE)[1:num_genes], ]
    # colnames(rna_mat) = sapply(colnames(rna_mat), function(x){
    #     return(unlist(strsplit(x, ".", T))[1])})
    return(rna_mat)
}


get_cor = function(chip_df, rna_df){
    rna_df = rna_df[,colnames(chip_df)]
    chip_df = chip_df[,colnames(rna_df)]
    rna_df_sample = rna_df[sample(1:nrow(rna_df), 200), ]
    chip_df_sample = chip_df[sample(1:nrow(chip_df), 100), ]
    cor_df_ref = apply(chip_df_sample, 1, function(chip_vals){
        chip_vals = (chip_vals - min(chip_vals)) / (max(chip_vals) - min(chip_vals))
        cor_vals = apply(rna_df_sample, 1, function(rna_vals){
            rna_vals = (rna_vals - min(rna_vals)) / (max(rna_vals) - min(rna_vals))
            cor_obj = cor.test(rna_vals, chip_vals)
            pval = cor_obj$p.value
            cor_val = cor_obj[[4]]
            if(pval > 0.1){
                cor_val = NA
            }
            return(cor_val)
        })
        return(cor_vals)
    })
    min_val = min(abs(cor_df_ref), na.rm=TRUE)
    rna_df = apply(rna_df, 1, function(rna_vals){
        return( (rna_vals - min(rna_vals)) / (max(rna_vals) - min(rna_vals)) )})
    chip_df = apply(chip_df, 1, function(chip_vals){
        return( (chip_vals - min(chip_vals)) / (max(chip_vals) - min(chip_vals)) )})
    cor_df = cor(rna_df, chip_df)
    cor_df[abs(cor_df) < min_val] = NA
    cor_df = as.data.frame(t(cor_df))
    return(cor_df)
}


# Assuming columns in job_idx_df include JobID, OutDir, ChipPath, Chrom, TF


if(file.exists(out_path)){
    stop(sprintf("%s already exists, exiting!\n", out_path))
}else{
    cat(sprintf("Started working on %s at %s\n", out_path, paste(Sys.time())))
}

chip_df = read.csv(chip_path, header=T, sep="\t", row.names=1, check.names=FALSE)
colnames(chip_df) = as.character(sapply(colnames(chip_df), function(cell_name){
    name_parts = unlist(strsplit(cell_name, ".", T))
    return(paste(name_parts[1:(length(name_parts) - 1)], collapse="."))}))
chip_idx = as.numeric(rownames(chip_df))
chrom_st_end = paste(chrom, ".", chip_idx * window, ".", ( (chip_idx * window) + window), sep="")
rownames(chip_df) = chrom_st_end


rna_df = get_rna_df(rna_path, colnames(chip_df), num_genes)
chip_df = chip_df[, colnames(rna_df)]

chip_df = chip_df[apply(chip_df, 1, sum) > 0, ]

if(ncol(chip_df) < 4){
    stop(sprintf("Found only %d matched cell types\n", ncol(chip_df)))
}
cor_df = get_cor(chip_df, rna_df)


gzlink = gzfile(out_path, "w")
write.table(cor_df, file=gzlink, quote=F, row.names=TRUE, sep="\t")
close(gzlink)
cat(sprintf("Done working on %s at %s\n", out_path, paste(Sys.time())))
