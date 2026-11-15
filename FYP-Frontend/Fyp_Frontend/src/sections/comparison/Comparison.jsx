import React from "react";
import { Scan, Sparkles, CheckCircle2, XCircle } from "lucide-react";
import MRIScan from "../../assets/images/MRISCAN.png";

export default function Comparison() {
  return (
    <section className="py-24 bg-gray-50 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[var(--medical-red)] to-transparent opacity-30" />

      <div className="max-w-6xl mx-auto px-6">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-red-50 border border-red-100 text-[var(--medical-red)] px-4 py-1.5 rounded-full text-sm font-semibold mb-4">
            <Scan className="w-3.5 h-3.5" />
            AI Enhancement
          </div>
          <h2 className="text-4xl font-extrabold text-gray-900 mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
            MRI Scan <span className="gradient-text">Analysis Comparison</span>
          </h2>
          <div className="section-divider" />
          <p className="text-gray-500 max-w-2xl mx-auto text-base leading-relaxed">
            See the difference AI-assisted analysis makes in detecting and tracking brain tumors with greater accuracy and confidence.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Without AI */}
          <div className="medical-card overflow-hidden group">
            <div className="relative overflow-hidden bg-gray-900 h-64">
              <img
                src={MRIScan}
                alt="Standard MRI Scan"
                className="w-full h-full object-cover opacity-70 group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-gray-900/80 to-transparent" />
              <div className="absolute top-4 left-4">
                <span className="bg-gray-700 text-gray-200 text-xs font-bold px-3 py-1.5 rounded-full uppercase tracking-wide">
                  Standard View
                </span>
              </div>
              <div className="absolute bottom-4 left-4 right-4">
                <div className="flex items-center gap-2 text-white">
                  <XCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                  <span className="text-sm font-medium">Subtle tumor signs may be overlooked</span>
                </div>
              </div>
            </div>
            <div className="p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-3" style={{ fontFamily: 'Sora, sans-serif' }}>
                Without AI Analysis
              </h3>
              <ul className="space-y-2">
                {[
                  "Manual interpretation required",
                  "Subtle lesions easily missed",
                  "No quantitative measurements",
                  "Time-intensive review process",
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-500">
                    <span className="w-1.5 h-1.5 bg-gray-300 rounded-full flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* With AI */}
          <div className="medical-card overflow-hidden group border-2 border-[var(--medical-red)]/20">
            <div className="relative overflow-hidden bg-gray-900 h-64">
              <img
                src={MRIScan}
                alt="AI-Enhanced MRI Scan"
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                style={{ filter: 'hue-rotate(180deg) saturate(1.5) brightness(1.1)' }}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-gray-900/80 to-transparent" />
              <div className="absolute top-4 left-4">
                <span className="bg-[var(--medical-red)] text-white text-xs font-bold px-3 py-1.5 rounded-full uppercase tracking-wide flex items-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  AI Enhanced
                </span>
              </div>
              {/* Simulated detection overlay */}
              <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 border-2 border-yellow-400 rounded-full opacity-80 animate-pulse" />
              <div className="absolute bottom-4 left-4 right-4">
                <div className="flex items-center gap-2 text-white">
                  <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                  <span className="text-sm font-medium">Tumor regions clearly highlighted</span>
                </div>
              </div>
            </div>
            <div className="p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-3" style={{ fontFamily: 'Sora, sans-serif' }}>
                With AI Analysis
              </h3>
              <ul className="space-y-2">
                {[
                  "Automated tumor segmentation",
                  "Precise boundary detection",
                  "Quantitative volume measurements",
                  "Results in under 60 seconds",
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                    <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
