import React from 'react';
import HeroImg from '../../assets/images/Hero.jpg';
import { Link } from 'react-router-dom';
import { ArrowRight, Brain, ShieldCheck, Activity, Zap, Users, Award } from 'lucide-react';

const stats = [
  { value: "98.2%", label: "Detection Accuracy", icon: <Brain className="w-4 h-4" /> },
  { value: "< 60s", label: "Analysis Time", icon: <Zap className="w-4 h-4" /> },
  { value: "500+", label: "Medical Professionals", icon: <Users className="w-4 h-4" /> },
  { value: "ISO 27001", label: "Data Security", icon: <ShieldCheck className="w-4 h-4" /> },
];

export default function Hero() {
  return (
    <section className="relative bg-gradient-to-br from-gray-50 via-white to-red-50 pt-28 pb-20 overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-gradient-to-bl from-red-100/40 to-transparent rounded-full -translate-y-1/3 translate-x-1/4 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-gradient-to-tr from-blue-50/60 to-transparent rounded-full translate-y-1/3 -translate-x-1/4 pointer-events-none" />

      {/* Floating medical cross icons */}
      <div className="absolute top-24 left-10 w-8 h-8 text-red-200 opacity-40 rotate-12 pointer-events-none">
        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M10 2h4v6h6v4h-6v6h-4v-6H4v-4h6z"/></svg>
      </div>
      <div className="absolute bottom-20 right-16 w-6 h-6 text-blue-200 opacity-40 -rotate-12 pointer-events-none">
        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M10 2h4v6h6v4h-6v6h-4v-6H4v-4h6z"/></svg>
      </div>

      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-col lg:flex-row items-center gap-16">

          {/* Left Content */}
          <div className="flex-1 text-center lg:text-left fade-in-up">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 bg-red-50 border border-red-100 text-[var(--medical-red)] px-4 py-1.5 rounded-full text-sm font-semibold mb-6">
              <span className="w-2 h-2 bg-[var(--medical-red)] rounded-full animate-pulse" />
              AI-Powered Neuro-Oncology Platform
            </div>

            <h1 className="text-5xl lg:text-6xl font-extrabold text-gray-900 leading-tight mb-6" style={{ fontFamily: 'Sora, sans-serif' }}>
              Detect Brain Tumors{" "}
              <span className="relative">
                <span className="gradient-text">with AI Precision</span>
                <svg className="absolute -bottom-2 left-0 w-full" viewBox="0 0 300 12" fill="none">
                  <path d="M2 10 Q75 2 150 8 Q225 14 298 6" stroke="#c0392b" strokeWidth="2.5" strokeLinecap="round" fill="none" opacity="0.4"/>
                </svg>
              </span>
            </h1>

            <p className="text-lg text-gray-600 leading-relaxed mb-8 max-w-xl mx-auto lg:mx-0">
              Upload MRI scans to detect gliomas, predict survival rates, and track tumor progression over time — all powered by advanced deep learning models.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start mb-12">
              <Link to="/auth">
                <button className="btn-shine flex items-center justify-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white px-8 py-3.5 rounded-xl font-semibold text-base shadow-lg shadow-red-200 transition-all duration-200 cursor-pointer">
                  Start Analysis
                  <ArrowRight className="w-4 h-4" />
                </button>
              </Link>
              <Link to="/howtouse">
                <button className="flex items-center justify-center gap-2 bg-white border-2 border-gray-200 hover:border-[var(--medical-red)] text-gray-700 hover:text-[var(--medical-red)] px-8 py-3.5 rounded-xl font-semibold text-base transition-all duration-200 cursor-pointer">
                  <Activity className="w-4 h-4" />
                  See How It Works
                </button>
              </Link>
            </div>

            {/* Trust indicators */}
            <div className="flex items-center gap-3 justify-center lg:justify-start text-sm text-gray-500">
              <ShieldCheck className="w-4 h-4 text-green-500" />
              <span>HIPAA Compliant</span>
              <span className="w-1 h-1 bg-gray-300 rounded-full" />
              <Award className="w-4 h-4 text-blue-500" />
              <span>Clinically Validated</span>
              <span className="w-1 h-1 bg-gray-300 rounded-full" />
              <span>256-bit Encryption</span>
            </div>
          </div>

          {/* Right Image */}
          <div className="flex-1 relative">
            <div className="relative rounded-2xl overflow-hidden shadow-2xl shadow-red-100">
              <img
                src={HeroImg}
                alt="MRI Brain Scan AI Analysis"
                className="w-full h-auto object-cover"
              />
              {/* Overlay gradient */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />

              {/* Floating analysis card */}
              <div className="absolute bottom-4 left-4 right-4 glass-card rounded-xl p-4 shadow-lg">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <Brain className="w-5 h-5 text-green-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-gray-700">AI Analysis Complete</p>
                    <p className="text-xs text-gray-500 truncate">Glioblastoma detected — 94.7% confidence</p>
                  </div>
                  <span className="flex-shrink-0 bg-green-100 text-green-700 text-xs font-bold px-2 py-1 rounded-full">Live</span>
                </div>
              </div>
            </div>

            {/* Floating stat badge */}
            <div className="absolute -top-4 -right-4 bg-white rounded-2xl shadow-xl p-4 border border-red-50">
              <p className="text-3xl font-extrabold text-[var(--medical-red)]" style={{ fontFamily: 'Sora, sans-serif' }}>98.2%</p>
              <p className="text-xs text-gray-500 font-medium">Detection Rate</p>
            </div>
          </div>
        </div>

        {/* Stats bar */}
        <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((stat, i) => (
            <div key={i} className="medical-card p-5 text-center">
              <div className="flex items-center justify-center gap-1.5 text-[var(--medical-red)] mb-2">
                {stat.icon}
                <span className="text-2xl font-extrabold" style={{ fontFamily: 'Sora, sans-serif' }}>{stat.value}</span>
              </div>
              <p className="text-xs text-gray-500 font-medium">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
