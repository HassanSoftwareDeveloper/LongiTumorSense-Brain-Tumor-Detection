import React from "react";
import { Facebook, Twitter, Linkedin, Mail, Phone, MapPin, Heart, Brain } from "lucide-react";
import { Link } from "react-router-dom";

export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="bg-gradient-to-br from-slate-900 via-gray-900 to-slate-900 text-white">
      {/* Top accent */}
      <div className="h-1 w-full bg-gradient-to-r from-[var(--medical-red)] via-rose-400 to-[var(--medical-red)]" />

      <div className="max-w-7xl mx-auto px-6 py-14">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10">

          {/* Brand */}
          <div className="md:col-span-1">
            <div className="flex items-center gap-2.5 mb-4">
              <div className="w-9 h-9 bg-[var(--medical-red)] rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="font-bold text-base" style={{ fontFamily: 'Sora, sans-serif' }}>LongiTumor<span className="text-red-400">Sense</span></div>
                <div className="text-gray-400 text-[10px] uppercase tracking-widest">AI Medical Platform</div>
              </div>
            </div>
            <p className="text-gray-400 text-sm leading-relaxed mb-5">
              AI-powered brain tumor detection using multi-modal and longitudinal MRI data. Secure. Accurate. Fast.
            </p>
            <div className="flex gap-3">
              {[
                { icon: <Facebook size={16} />, href: "#" },
                { icon: <Twitter size={16} />, href: "#" },
                { icon: <Linkedin size={16} />, href: "#" },
                { icon: <Mail size={16} />, href: "mailto:info@longitumor.ai" },
              ].map((social, i) => (
                <a
                  key={i}
                  href={social.href}
                  className="w-8 h-8 bg-white/10 hover:bg-[var(--medical-red)] rounded-lg flex items-center justify-center transition-colors duration-200"
                >
                  {social.icon}
                </a>
              ))}
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-sm font-bold uppercase tracking-widest text-gray-300 mb-4">Platform</h3>
            <ul className="space-y-2.5">
              {[
                { to: "/", label: "Home" },
                { to: "/about", label: "About Us" },
                { to: "/howtouse", label: "How to Use" },
                { to: "/auth", label: "Login / Sign Up" },
                { to: "/dashboard", label: "Dashboard" },
              ].map((link) => (
                <li key={link.to}>
                  <Link to={link.to} className="text-gray-400 hover:text-white text-sm transition-colors duration-200 flex items-center gap-1.5 group">
                    <span className="w-1 h-1 bg-[var(--medical-red)] rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-sm font-bold uppercase tracking-widest text-gray-300 mb-4">Resources</h3>
            <ul className="space-y-2.5">
              {[
                { to: "/contact", label: "Contact Support" },
                { to: "/howtouse", label: "Documentation" },
                { to: "/about", label: "Research & Team" },
                { to: "/reports", label: "Patient Reports" },
              ].map((link) => (
                <li key={link.to}>
                  <Link to={link.to} className="text-gray-400 hover:text-white text-sm transition-colors duration-200 flex items-center gap-1.5 group">
                    <span className="w-1 h-1 bg-[var(--medical-red)] rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-sm font-bold uppercase tracking-widest text-gray-300 mb-4">Contact</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-3 text-gray-400 text-sm">
                <Mail className="w-4 h-4 text-[var(--medical-red)] flex-shrink-0 mt-0.5" />
                <span>support@longitumorsense.com</span>
              </li>
              <li className="flex items-start gap-3 text-gray-400 text-sm">
                <Phone className="w-4 h-4 text-[var(--medical-red)] flex-shrink-0 mt-0.5" />
                <span>+92 300 1234567</span>
              </li>
              <li className="flex items-start gap-3 text-gray-400 text-sm">
                <MapPin className="w-4 h-4 text-[var(--medical-red)] flex-shrink-0 mt-0.5" />
                <span>123 Medical Research Park, Islamabad, Pakistan</span>
              </li>
            </ul>

            {/* Compliance badges */}
            <div className="mt-5 flex flex-wrap gap-2">
              {["HIPAA", "ISO 27001", "GDPR"].map((badge) => (
                <span key={badge} className="bg-white/10 text-gray-300 text-[10px] font-bold px-2.5 py-1 rounded uppercase tracking-wide">
                  {badge}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-12 pt-6 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-gray-500 text-sm flex items-center gap-1.5">
            © {year} LongiTumor AI. Made with <Heart className="w-3.5 h-3.5 text-red-400 fill-red-400" /> for better healthcare.
          </p>
          <div className="flex gap-6 text-gray-500 text-xs">
            <span className="hover:text-white cursor-pointer transition-colors">Privacy Policy</span>
            <span className="hover:text-white cursor-pointer transition-colors">Terms of Service</span>
            <span className="hover:text-white cursor-pointer transition-colors">Cookie Policy</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
