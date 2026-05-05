import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { CloudUpload, ImageIcon, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  onFile: (file: File) => void;
  preview?: string | null;
  className?: string;
}

export function Dropzone({ onFile, preview, className }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => { if (accepted[0]) onFile(accepted[0]); },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } =
    useDropzone({
      accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"] },
      maxFiles: 1,
      maxSize: 12 * 1024 * 1024,
      onDrop,
    });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative group cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden",
        isDragReject
          ? "border-red-500/60 bg-red-500/5"
          : isDragActive
          ? "border-violet-400/70 bg-violet-500/10 scale-[1.01]"
          : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]",
        className
      )}
    >
      <input {...getInputProps()} />

      {preview ? (
        /* Show image preview */
        <div className="relative aspect-video w-full">
          <img
            src={preview}
            alt="preview"
            className="w-full h-full object-contain"
          />
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 text-white text-sm font-medium">
            <X className="h-4 w-4" /> Replace image
          </div>
        </div>
      ) : (
        /* Empty state */
        <div className="flex flex-col items-center justify-center gap-4 px-6 py-14">
          <div
            className={cn(
              "h-16 w-16 rounded-2xl flex items-center justify-center transition-all duration-300",
              isDragActive
                ? "bg-violet-500/30 scale-110"
                : "bg-white/5 group-hover:bg-violet-500/15 group-hover:scale-105"
            )}
          >
            {isDragActive ? (
              <ImageIcon className="h-7 w-7 text-violet-300" />
            ) : (
              <CloudUpload className="h-7 w-7 text-foreground/40 group-hover:text-violet-300 transition-colors duration-200" />
            )}
          </div>

          <div className="text-center">
            <p className="text-sm font-medium text-foreground/80">
              {isDragReject
                ? "File type not supported"
                : isDragActive
                ? "Drop it here!"
                : "Drag & drop an image, or click to browse"}
            </p>
            <p className="mt-1 text-xs text-foreground/40">
              JPG · PNG · WEBP · BMP  ·  up to 12 MB
            </p>
          </div>

          <div className="flex items-center gap-3 text-[10px] text-foreground/30">
            {["Syringe", "Gloves", "Gauze", "Vials", "Masks"].map((s) => (
              <span
                key={s}
                className="px-2 py-1 rounded-full border border-white/8 bg-white/3"
              >
                {s}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
