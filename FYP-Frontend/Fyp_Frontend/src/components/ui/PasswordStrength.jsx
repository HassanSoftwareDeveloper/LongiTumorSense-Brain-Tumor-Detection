import React from "react";
import { Check, X } from "lucide-react";

// Rules every password must satisfy
export const PASSWORD_RULES = [
  { id: "length",  label: "At least 8 characters",       test: (p) => p.length >= 8 },
  { id: "upper",   label: "At least 1 uppercase letter",  test: (p) => /[A-Z]/.test(p) },
  { id: "special", label: "At least 1 special character", test: (p) => /[^A-Za-z0-9]/.test(p) },
];

/**
 * Returns null when valid, or an error string when invalid.
 * Use this before submitting any form.
 */
export function validatePassword(password) {
  for (const rule of PASSWORD_RULES) {
    if (!rule.test(password)) return `Password must have ${rule.label.toLowerCase()}.`;
  }
  return null;
}

/**
 * Visual strength indicator + checklist shown below a password field.
 * Only renders when `password` is non-empty.
 */
export default function PasswordStrength({ password }) {
  if (!password) return null;

  const passed = PASSWORD_RULES.filter((r) => r.test(password)).length;
  const total  = PASSWORD_RULES.length;

  const strengthLabel = passed === total ? "Strong" : passed === 2 ? "Medium" : "Weak";
  const barColor =
    passed === total ? "bg-green-500" :
    passed === 2     ? "bg-amber-400" :
                       "bg-red-400";

  return (
    <div className="mt-2 space-y-2">
      {/* Strength bar */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden flex gap-0.5">
          {PASSWORD_RULES.map((_, i) => (
            <div
              key={i}
              className={`flex-1 rounded-full transition-all duration-300 ${
                i < passed ? barColor : "bg-gray-200"
              }`}
            />
          ))}
        </div>
        <span className={`text-[11px] font-semibold w-12 text-right ${
          passed === total ? "text-green-600" :
          passed === 2     ? "text-amber-500" :
                             "text-red-500"
        }`}>
          {strengthLabel}
        </span>
      </div>

      {/* Rule checklist */}
      <ul className="space-y-1">
        {PASSWORD_RULES.map((rule) => {
          const ok = rule.test(password);
          return (
            <li key={rule.id} className="flex items-center gap-1.5">
              <span className={`flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center ${
                ok ? "bg-green-100" : "bg-red-50"
              }`}>
                {ok
                  ? <Check className="w-2.5 h-2.5 text-green-600" />
                  : <X    className="w-2.5 h-2.5 text-red-400"   />
                }
              </span>
              <span className={`text-[11px] ${ok ? "text-green-600" : "text-gray-400"}`}>
                {rule.label}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
