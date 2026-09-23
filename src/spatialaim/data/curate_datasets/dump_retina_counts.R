# Dump the undamaged-P60 raw counts of the scRetinaDB mouse light-damage object.
#
#   Rscript dump_retina_counts.R <object.qs> <out_dir>
#
# Called by curate_06_mouse_retina.py, which needs the counts of dataset
# 06_mouse-retina. They are only distributed as an R `qs`-serialised Seurat
# object (scRetinaDB accession GitHub-jiewwwang), and `qs` has no Python reader,
# so this one step runs in R; everything else stays in Python.
#
# Writes into <out_dir>: counts.mtx (MatrixMarket, features x cells),
# genes.txt and cells.txt (the corresponding names, one per line).
#
# The object is Seurat v5, so the counts live in assays$RNA@layers$counts, not
# in the v4 @counts slot. Slots are reached with attr() so that SeuratObject is
# not required — only qs and Matrix.

suppressMessages({
  library(qs)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: dump_retina_counts.R <object.qs> <out_dir>")
object_path <- args[1]
out_dir <- args[2]

obj <- qread(object_path)

metadata <- attr(obj, "meta.data")
if (!"status" %in% colnames(metadata)) {
  stop("the object has no 'status' column — cannot select the undamaged samples")
}
# status is mmP (undamaged P60 controls) or mmLD (light-damaged); the four P60
# replicates are the reference, the damaged time course is not.
keep <- which(metadata$status == "mmP")
cat("cells:", nrow(metadata), "-> undamaged P60:", length(keep), "\n")

rna <- attr(obj, "assays")[["RNA"]]
counts <- attr(rna, "layers")[["counts"]]
if (is.null(counts)) stop("no 'counts' layer in assay RNA — this is not a raw object")
dimnames(counts) <- list(
  attr(attr(rna, "features"), "dimnames")[[1]],
  attr(attr(rna, "cells"), "dimnames")[[1]]
)

subset <- counts[, keep, drop = FALSE]
cat("matrix:", nrow(subset), "x", ncol(subset), "| nonzeros:", length(subset@x), "\n")
if (any(subset@x != round(subset@x))) {
  stop("the counts layer is not integral — it is not a raw count matrix")
}

invisible(writeMM(subset, file.path(out_dir, "counts.mtx")))
writeLines(rownames(subset), file.path(out_dir, "genes.txt"))
writeLines(colnames(subset), file.path(out_dir, "cells.txt"))
cat("written to", out_dir, "\n")
