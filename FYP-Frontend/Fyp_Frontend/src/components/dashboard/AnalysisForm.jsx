import React from "react";
import { Upload, Scan, Cpu, CheckCircle2, FileImage, Loader2, Brain } from "lucide-react";

export default function AnalysisForm({
  exploreFile,
  handleExploreUpload,
  isAnalyzing,
  handleRunAnalysis,
  compareFirstFile,
  handleCompareFirstUpload,
  compareLatestFile,
  handleCompareLatestUpload,
}) {
  return (
    <div className="flex-1 p-4 sm:p-6 overflow-y-auto bg-gray-50">
      {/* Header */}
      <div className="mb-5">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-8 h-8 bg-[var(--medical-red)] rounded-lg flex items-center justify-center flex-shrink-0">
            <Brain className="w-4 h-4 text-white" />
          </div>
          <h1 className="text-lg sm:text-xl font-extrabold text-gray-900" style={{ fontFamily: 'Sora, sans-serif' }}>
            MRI Analysis
          </h1>
        </div>
        <p className="text-sm text-gray-500 ml-10">
          Upload MRI scans for AI-powered tumor detection
        </p>
      </div>

      <div className="space-y-4">
        {/* ── Explore — Single MRI ── */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          <div className="p-4 sm:p-5 border-b border-gray-50">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 sm:w-10 sm:h-10 bg-red-50 rounded-xl flex items-center justify-center flex-shrink-0">
                <Scan className="w-4 h-4 sm:w-5 sm:h-5 text-[var(--medical-red)]" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>
                  Explore — Single MRI
                </h3>
                <p className="text-xs text-gray-400">Detect tumor from one MRI scan</p>
              </div>
            </div>
          </div>

          <div className="p-4 sm:p-5">
            <label className="block cursor-pointer">
              <div className={`border-2 border-dashed rounded-xl p-5 text-center transition-all duration-200 ${
                exploreFile
                  ? "border-green-300 bg-green-50"
                  : "border-gray-200 hover:border-[var(--medical-red)] hover:bg-red-50/30"
              }`}>
                {exploreFile ? (
                  <div className="flex items-center justify-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <div className="text-left min-w-0">
                      <p className="text-sm font-semibold text-green-700">File Ready</p>
                      <p className="text-xs text-green-600 truncate max-w-[180px] sm:max-w-[240px]">
                        {exploreFile.name}
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    <FileImage className="w-8 h-8 text-gray-300 mx-auto mb-2" />
                    <p className="text-sm font-medium text-gray-500">Tap to upload MRI scan</p>
                    <p className="text-xs text-gray-400 mt-1">DICOM, NIfTI, PNG, JPEG</p>
                  </>
                )}
                <input type="file" onChange={handleExploreUpload} className="hidden" />
              </div>
            </label>

            <button
              onClick={() => handleRunAnalysis("explore")}
              disabled={isAnalyzing || !exploreFile}
              className={`mt-4 w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                exploreFile && !isAnalyzing
                  ? "bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white shadow-sm shadow-red-100 cursor-pointer"
                  : "bg-gray-100 text-gray-400 cursor-not-allowed"
              }`}
            >
              {isAnalyzing ? (
                <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
              ) : (
                <><Cpu className="w-4 h-4" /> Run Explore Analysis</>
              )}
            </button>
          </div>
        </div>

        {/* ── Compare — Two MRIs ── */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          <div className="p-4 sm:p-5 border-b border-gray-50">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 sm:w-10 sm:h-10 bg-blue-50 rounded-xl flex items-center justify-center flex-shrink-0">
                <Upload className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>
                  Compare — Two MRIs
                </h3>
                <p className="text-xs text-gray-400">Track tumor progression over time</p>
              </div>
            </div>
          </div>

          <div className="p-4 sm:p-5 space-y-4">
            {/* Two upload zones side-by-side on sm+, stacked on xs */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* First MRI */}
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">
                  1st Timestamp
                </label>
                <label className="block cursor-pointer">
                  <div className={`border-2 border-dashed rounded-xl p-4 text-center transition-all duration-200 ${
                    compareFirstFile
                      ? "border-green-300 bg-green-50"
                      : "border-gray-200 hover:border-blue-300 hover:bg-blue-50/30"
                  }`}>
                    {compareFirstFile ? (
                      <div className="flex items-center justify-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4 text-green-600 flex-shrink-0" />
                        <p className="text-xs font-semibold text-green-700 truncate max-w-[100px]">
                          {compareFirstFile.name}
                        </p>
                      </div>
                    ) : (
                      <>
                        <FileImage className="w-6 h-6 text-gray-300 mx-auto mb-1" />
                        <p className="text-xs text-gray-400">Upload first scan</p>
                      </>
                    )}
                    <input type="file" onChange={handleCompareFirstUpload} className="hidden" />
                  </div>
                </label>
              </div>

              {/* Latest MRI */}
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">
                  Latest Timestamp
                </label>
                <label className="block cursor-pointer">
                  <div className={`border-2 border-dashed rounded-xl p-4 text-center transition-all duration-200 ${
                    compareLatestFile
                      ? "border-green-300 bg-green-50"
                      : "border-gray-200 hover:border-blue-300 hover:bg-blue-50/30"
                  }`}>
                    {compareLatestFile ? (
                      <div className="flex items-center justify-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4 text-green-600 flex-shrink-0" />
                        <p className="text-xs font-semibold text-green-700 truncate max-w-[100px]">
                          {compareLatestFile.name}
                        </p>
                      </div>
                    ) : (
                      <>
                        <FileImage className="w-6 h-6 text-gray-300 mx-auto mb-1" />
                        <p className="text-xs text-gray-400">Upload latest scan</p>
                      </>
                    )}
                    <input type="file" onChange={handleCompareLatestUpload} className="hidden" />
                  </div>
                </label>
              </div>
            </div>

            <button
              onClick={() => handleRunAnalysis("compare")}
              disabled={isAnalyzing || !compareFirstFile || !compareLatestFile}
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                compareFirstFile && compareLatestFile && !isAnalyzing
                  ? "bg-blue-600 hover:bg-blue-700 text-white shadow-sm shadow-blue-100 cursor-pointer"
                  : "bg-gray-100 text-gray-400 cursor-not-allowed"
              }`}
            >
              {isAnalyzing ? (
                <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
              ) : (
                <><Cpu className="w-4 h-4" /> Run Compare Analysis</>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
