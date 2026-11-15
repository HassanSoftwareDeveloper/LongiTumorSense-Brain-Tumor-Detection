import React from "react";
import { UserCheck, Upload, Cpu, FileCheck, ChevronRight } from "lucide-react";

const steps = [
  {
    step: "01",
    icon: <UserCheck className="w-7 h-7" />,
    title: "Sign Up & Verification",
    description: "Register and get verified by the admin to ensure only authorized medical practitioners can access the platform.",
    color: "from-red-500 to-rose-600",
    bg: "bg-red-50",
    iconColor: "text-red-600",
  },
  {
    step: "02",
    icon: <Upload className="w-7 h-7" />,
    title: "Upload MRI Scans",
    description: "Securely upload single or multiple MRI scans for tumor detection, progression tracking, and prediction analysis.",
    color: "from-orange-500 to-amber-600",
    bg: "bg-orange-50",
    iconColor: "text-orange-600",
  },
  {
    step: "03",
    icon: <Cpu className="w-7 h-7" />,
    title: "AI Processing",
    description: "Advanced deep learning models analyze scans using multi-modal and longitudinal data for highly accurate results.",
    color: "from-blue-500 to-indigo-600",
    bg: "bg-blue-50",
    iconColor: "text-blue-600",
  },
  {
    step: "04",
    icon: <FileCheck className="w-7 h-7" />,
    title: "Get Reports",
    description: "Receive a detailed tumor detection report with survival estimates, risk assessment, and personalized treatment suggestions.",
    color: "from-green-500 to-emerald-600",
    bg: "bg-green-50",
    iconColor: "text-green-600",
  },
];

export default function HowItWorks() {
  return (
    <section className="py-24 bg-white relative overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-[0.03]" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c0392b' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
      }} />

      <div className="max-w-7xl mx-auto px-6 relative">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-red-50 border border-red-100 text-[var(--medical-red)] px-4 py-1.5 rounded-full text-sm font-semibold mb-4">
            <Cpu className="w-3.5 h-3.5" />
            Streamlined Workflow
          </div>
          <h2 className="text-4xl font-extrabold text-gray-900 mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
            How <span className="gradient-text">LongiTumorSense</span> Works
          </h2>
          <div className="section-divider" />
          <p className="text-gray-500 max-w-2xl mx-auto text-base leading-relaxed">
            A secure, accurate, and efficient workflow — from verification to advanced AI analysis — helping medical professionals make informed decisions.
          </p>
        </div>

        {/* Steps */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative">
          {/* Connector line (desktop) */}
          <div className="hidden lg:block absolute top-12 left-[12.5%] right-[12.5%] h-0.5 bg-gradient-to-r from-red-200 via-blue-200 to-green-200 z-0" />

          {steps.map((step, index) => (
            <div key={index} className="relative z-10 group">
              <div className="medical-card p-6 h-full flex flex-col items-center text-center">
                {/* Step number */}
                <div className="text-xs font-bold text-gray-300 mb-3 tracking-widest">{step.step}</div>

                {/* Icon circle */}
                <div className={`w-16 h-16 ${step.bg} rounded-2xl flex items-center justify-center mb-4 ${step.iconColor} group-hover:scale-110 transition-transform duration-300 shadow-sm`}>
                  {step.icon}
                </div>

                <h3 className="text-base font-bold text-gray-800 mb-2" style={{ fontFamily: 'Sora, sans-serif' }}>
                  {step.title}
                </h3>
                <p className="text-gray-500 text-sm leading-relaxed">{step.description}</p>

                {/* Arrow connector (mobile) */}
                {index < steps.length - 1 && (
                  <div className="lg:hidden mt-4 text-gray-300">
                    <ChevronRight className="w-5 h-5 mx-auto rotate-90" />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="mt-16 text-center">
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-red-50 to-rose-50 border border-red-100 rounded-2xl px-8 py-4">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-gray-700 text-sm font-medium">
              Average analysis time: <strong className="text-[var(--medical-red)]">under 60 seconds</strong> per scan
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
