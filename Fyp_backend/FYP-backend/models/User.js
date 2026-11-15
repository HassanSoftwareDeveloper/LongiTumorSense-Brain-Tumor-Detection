import mongoose from "mongoose";

const userSchema = new mongoose.Schema({
  fullName: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  organization: { type: String, required: true },
  specialty: { type: String, required: true },
  certificate: { type: String, required: true }, // file path or cloud URL
  status: { type: String, enum: ["Pending", "Approved", "Rejected"], default: "Pending" },
  resetToken: { type: String, default: null }, // hashed reset token
  resetTokenExpiry: { type: Date, default: null } // expiry time for reset token
}, { timestamps: true });

export default mongoose.model("User", userSchema);
