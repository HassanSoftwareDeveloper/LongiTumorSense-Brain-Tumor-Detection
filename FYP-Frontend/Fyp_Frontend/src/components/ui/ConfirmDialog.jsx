import * as React from "react";
import { Dialog, DialogContent, DialogTitle } from "@radix-ui/react-dialog";
import { AlertTriangle, X } from "lucide-react";

export default function ConfirmDialog({
  open,
  onOpenChange,
  onConfirm,
  onCancel,
  title = "Are you sure?",
  description = "This action cannot be undone.",
  confirmText = "Delete",
  cancelText = "Cancel",
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="fixed inset-0 flex items-center justify-center z-50 bg-black/50 backdrop-blur-sm p-4">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden">
          {/* Header */}
          <div className="bg-red-50 border-b border-red-100 p-5 flex items-center gap-3">
            <div className="w-10 h-10 bg-red-100 rounded-xl flex items-center justify-center flex-shrink-0">
              <AlertTriangle className="w-5 h-5 text-red-600" />
            </div>
            <DialogTitle className="text-base font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>
              {title}
            </DialogTitle>
          </div>

          <div className="p-5">
            <p className="text-sm text-gray-600 leading-relaxed mb-5">{description}</p>
            <div className="flex gap-3">
              <button
                onClick={onCancel}
                className="flex-1 py-2.5 border border-gray-200 rounded-xl text-sm font-semibold text-gray-600 hover:bg-gray-50 transition-colors cursor-pointer"
              >
                {cancelText}
              </button>
              <button
                onClick={onConfirm}
                className="flex-1 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-xl text-sm font-semibold transition-colors cursor-pointer shadow-sm shadow-red-100"
              >
                {confirmText}
              </button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
