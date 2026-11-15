import React from "react";
import { User, Brain, CheckCircle2, XCircle, Save, TrendingUp } from "lucide-react";

export default function PatientDetails({ selectedPatient, analysisResult, saveToHistory, handleSaveHistory }) {
  const confidencePct = analysisResult?.confidence
    ? (analysisResult.confidence * 100).toFixed(1)
    : null;

  return (
    <div className="w-full h-full flex-shrink-0 bg-white border-l border-gray-100 flex flex-col shadow-sm lg:w-72">
      {/* Header */}
      <div className="p-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-7 h-7 bg-blue-50 rounded-lg flex items-center justify-center">
            <User className="w-4 h-4 text-blue-600" />
          </div>
          <h2 className="text-sm font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>Patient Details</h2>
        </div>
        <p className="text-xs text-gray-400 ml-9">Selected patient info</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {selectedPatient ? (
          <div className="space-y-4">
            {/* Patient card */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {selectedPatient.name?.charAt(0)?.toUpperCase()}
                </div>
                <div>
                  <p className="font-bold text-gray-800 text-sm">{selectedPatient.name}</p>
                  <p className="text-xs text-gray-500">{selectedPatient.gender}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-white rounded-lg p-2 text-center">
                  <p className="text-lg font-extrabold text-blue-600" style={{ fontFamily: 'Sora, sans-serif' }}>{selectedPatient.age}</p>
                  <p className="text-[10px] text-gray-400 uppercase tracking-wide">Age</p>
                </div>
                <div className="bg-white rounded-lg p-2 text-center">
                  <p className="text-sm font-bold text-blue-600">{selectedPatient.gender}</p>
                  <p className="text-[10px] text-gray-400 uppercase tracking-wide">Gender</p>
                </div>
              </div>
            </div>

            {/* Analysis results */}
            {analysisResult && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-[var(--medical-red)]" />
                  <h3 className="text-sm font-bold text-gray-700" style={{ fontFamily: 'Sora, sans-serif' }}>Analysis Results</h3>
                </div>

                {/* Status */}
                <div className={`flex items-center gap-2 p-3 rounded-xl text-sm font-medium ${
                  analysisResult.success ? "bg-green-50 text-green-700 border border-green-100" : "bg-red-50 text-red-700 border border-red-100"
                }`}>
                  {analysisResult.success
                    ? <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                    : <XCircle className="w-4 h-4 flex-shrink-0" />
                  }
                  {analysisResult.success ? "Analysis Successful" : "Analysis Failed"}
                </div>

                {/* Tumor detection */}
                <div className="bg-gray-50 rounded-xl p-3 space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-500">Tumor Detected</span>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                      analysisResult.tumorDetected ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                    }`}>
                      {analysisResult.tumorDetected ? "Yes" : "No"}
                    </span>
                  </div>
                  {analysisResult.tumorType && (
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-500">Tumor Type</span>
                      <span className="text-xs font-semibold text-gray-700">{analysisResult.tumorType}</span>
                    </div>
                  )}
                  {confidencePct && (
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs text-gray-500">Confidence</span>
                        <span className="text-xs font-bold text-[var(--medical-red)]">{confidencePct}%</span>
                      </div>
                      <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-red-400 to-red-600 rounded-full transition-all duration-1000"
                          style={{ width: `${confidencePct}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Report summary */}
                <div className="bg-gray-50 rounded-xl p-3 space-y-1.5">
                  <p className="text-xs font-semibold text-gray-600 mb-2">Report Summary</p>
                  <div className="flex justify-between">
                    <span className="text-xs text-gray-400">Type</span>
                    <span className="text-xs font-medium text-gray-700">{analysisResult.type || "Unknown"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-gray-400">Patient</span>
                    <span className="text-xs font-medium text-gray-700 truncate ml-2">{selectedPatient.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-gray-400">Date</span>
                    <span className="text-xs font-medium text-gray-700">{new Date().toLocaleDateString()}</span>
                  </div>
                </div>

                {/* Save button */}
                {saveToHistory && (
                  <button
                    onClick={handleSaveHistory}
                    className="w-full flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white py-2.5 rounded-xl text-sm font-semibold transition-colors cursor-pointer"
                  >
                    <Save className="w-4 h-4" />
                    Save to History
                  </button>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <User className="w-8 h-8 text-gray-300" />
            </div>
            <p className="text-sm text-gray-400 font-medium">No patient selected</p>
            <p className="text-xs text-gray-300 mt-1">Select a patient from the list</p>
          </div>
        )}
      </div>
    </div>
  );
}
