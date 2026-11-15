import React, { useState } from "react";
import { Mail, Phone, MapPin, Send, Clock, MessageSquare, ArrowRight } from "lucide-react";
import toast, { Toaster } from "react-hot-toast";

export default function Contact() {
  const [form, setForm] = useState({ name: "", email: "", message: "" });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("https://formspree.io/f/mzddzawd", {
        method: "POST",
        headers: { Accept: "application/json" },
        body: new FormData(e.target),
      });
      setLoading(false);
      if (res.ok) {
        toast.success("Message sent successfully!");
        setForm({ name: "", email: "", message: "" });
      } else {
        toast.error("Submission failed. Please try again.");
      }
    } catch (err) {
      setLoading(false);
      toast.error("Submission failed. Please try again.");
    }
  };

  const inputClass = "medical-input w-full px-4 py-3 border border-gray-200 rounded-xl text-sm text-gray-800 placeholder-gray-400 bg-gray-50 focus:bg-white transition-all duration-200";

  const contactInfo = [
    {
      icon: <Mail className="w-5 h-5" />,
      label: "Email Us",
      value: "support@longitumorsense.com",
      sub: "We reply within 24 hours",
      color: "text-red-600",
      bg: "bg-red-50",
    },
    {
      icon: <Phone className="w-5 h-5" />,
      label: "Call Us",
      value: "+92 300 1234567",
      sub: "Mon–Fri, 9am–6pm PKT",
      color: "text-blue-600",
      bg: "bg-blue-50",
    },
    {
      icon: <MapPin className="w-5 h-5" />,
      label: "Visit Us",
      value: "123 Medical Research Park",
      sub: "Islamabad, Pakistan",
      color: "text-green-600",
      bg: "bg-green-50",
    },
    {
      icon: <Clock className="w-5 h-5" />,
      label: "Support Hours",
      value: "24/7 Online Support",
      sub: "Always here to help",
      color: "text-purple-600",
      bg: "bg-purple-50",
    },
  ];

  return (
    <div className="bg-gray-50 mt-16">
      <Toaster position="top-right" />

      {/* Hero */}
      <section className="relative bg-gradient-to-br from-slate-900 via-red-950 to-slate-900 py-20 text-center overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: `radial-gradient(circle at 30% 50%, rgba(192,57,43,0.8) 0%, transparent 50%)`
        }} />
        <div className="relative z-10 max-w-3xl mx-auto px-6">
          <div className="inline-flex items-center gap-2 bg-white/10 border border-white/20 text-white px-4 py-1.5 rounded-full text-sm font-semibold mb-6 backdrop-blur-sm">
            <MessageSquare className="w-3.5 h-3.5 text-red-300" />
            Get in Touch
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4" style={{ fontFamily: 'Sora, sans-serif' }}>
            Contact <span className="text-red-300">Our Team</span>
          </h1>
          <p className="text-gray-300 text-lg leading-relaxed">
            Have questions about LongiTumorSense? Need technical support? We're here to help you 24/7.
          </p>
        </div>
      </section>

      {/* Contact info cards */}
      <section className="max-w-6xl mx-auto px-6 -mt-8 relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {contactInfo.map((info, i) => (
            <div key={i} className="medical-card p-5 text-center">
              <div className={`w-10 h-10 ${info.bg} ${info.color} rounded-xl flex items-center justify-center mx-auto mb-3`}>
                {info.icon}
              </div>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wide mb-1">{info.label}</p>
              <p className="text-sm font-semibold text-gray-800 leading-tight">{info.value}</p>
              <p className="text-xs text-gray-400 mt-0.5">{info.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Main content */}
      <section className="py-16 max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-2 gap-12">

          {/* Form */}
          <div className="medical-card p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-red-50 rounded-xl flex items-center justify-center">
                <Send className="w-5 h-5 text-[var(--medical-red)]" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-800" style={{ fontFamily: 'Sora, sans-serif' }}>Send a Message</h2>
                <p className="text-xs text-gray-400">We'll get back to you within 24 hours</p>
              </div>
            </div>

            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1.5 uppercase tracking-wide">Your Name</label>
                <input
                  type="text"
                  name="name"
                  placeholder="Dr. John Smith"
                  value={form.name}
                  onChange={handleChange}
                  className={inputClass}
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1.5 uppercase tracking-wide">Email Address</label>
                <input
                  type="email"
                  name="email"
                  placeholder="doctor@hospital.com"
                  value={form.email}
                  onChange={handleChange}
                  className={inputClass}
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1.5 uppercase tracking-wide">Message</label>
                <textarea
                  placeholder="Describe your question or issue..."
                  name="message"
                  rows="5"
                  value={form.message}
                  onChange={handleChange}
                  className={inputClass + " resize-none"}
                  required
                />
              </div>
              <button
                type="submit"
                className="btn-shine w-full flex items-center justify-center gap-2 bg-[var(--medical-red)] hover:bg-[var(--medical-red-hover)] text-white font-semibold px-6 py-3.5 rounded-xl shadow-lg shadow-red-100 transition-all duration-200 cursor-pointer"
                disabled={loading}
              >
                {loading ? "Sending..." : "Send Message"}
                {!loading && <ArrowRight className="w-4 h-4" />}
              </button>
            </form>
          </div>

          {/* Info side */}
          <div className="flex flex-col gap-6">
            <div className="medical-card p-8">
              <h2 className="text-xl font-bold text-gray-800 mb-2" style={{ fontFamily: 'Sora, sans-serif' }}>Frequently Asked</h2>
              <p className="text-gray-500 text-sm mb-6">Quick answers to common questions</p>
              <div className="space-y-4">
                {[
                  {
                    q: "How do I get verified?",
                    a: "Submit your medical certificate during signup. Our admin team reviews and approves within 1–2 business days.",
                  },
                  {
                    q: "What MRI formats are supported?",
                    a: "We support DICOM, NIfTI, PNG, and JPEG formats. DICOM is recommended for best results.",
                  },
                  {
                    q: "Is patient data secure?",
                    a: "Yes. All data is encrypted with AES-256 and we are fully HIPAA compliant.",
                  },
                ].map((faq, i) => (
                  <div key={i} className="border-l-2 border-[var(--medical-red)] pl-4">
                    <p className="text-sm font-semibold text-gray-800 mb-1">{faq.q}</p>
                    <p className="text-xs text-gray-500 leading-relaxed">{faq.a}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Map placeholder */}
            <div className="medical-card p-6 bg-gradient-to-br from-red-50 to-rose-50 border border-red-100">
              <div className="flex items-center gap-3 mb-3">
                <MapPin className="w-5 h-5 text-[var(--medical-red)]" />
                <h3 className="font-bold text-gray-800 text-sm" style={{ fontFamily: 'Sora, sans-serif' }}>Our Location</h3>
              </div>
              <p className="text-sm text-gray-600">123 Medical Research Park</p>
              <p className="text-sm text-gray-600">Islamabad, Pakistan 44000</p>
              <div className="mt-4 h-32 bg-gradient-to-br from-gray-200 to-gray-300 rounded-xl flex items-center justify-center">
                <p className="text-gray-400 text-xs font-medium">Map View</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
