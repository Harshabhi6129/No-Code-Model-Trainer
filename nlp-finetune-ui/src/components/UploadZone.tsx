import React, { useState } from "react";
import { useDropzone }     from "react-dropzone";
import Papa                from "papaparse";

export default function UploadZone({
  onFile,
  disabled,
}: {
  onFile: (f: File) => void;
  disabled?: boolean;
}) {
  const [preview, setPreview] = useState<Record<string, unknown>[]>([]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "text/csv": [".csv"] },
    multiple: false,
    disabled,
    onDrop: (files) => {
      const f = files[0];
      if (!f) return;
      Papa.parse(f, {
        header: true,
        preview: 3,
        complete: (res) => setPreview(res.data as any),
      });
      onFile(f);
    },
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded text-center cursor-pointer px-4 py-6
                  ${disabled ? "opacity-40" : "hover:bg-gray-50"}`}
    >
      <input {...getInputProps()} />
      {isDragActive ? "Drop the CSV here…" : "Drag & drop CSV or click"}
      {preview.length > 0 && (
        <table className="mt-4 mx-auto text-xs border">
          <thead>
            <tr>
              {Object.keys(preview[0]).map((k) => (
                <th key={k} className="border px-1">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((r, i) => (
              <tr key={i}>
                {Object.values(r).map((v, j) => (
                  <td
                    key={j}
                    className="border px-1 truncate max-w-[6rem]"
                    title={String(v)}
                  >
                    {String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
