// import bcrypt from "bcryptjs";
// import jwt from "jsonwebtoken";
// import User from "../models/User.js";
// import sendEmail from "../utils/email.js";

// export const signupUser = async (req, res) => {
//   try {
//     const { fullName, email, password, organization, specialty } = req.body;
//     if (!req.file) return res.status(400).json({ message: "Certificate required" });

//     // check existing
//     const exists = await User.findOne({ email });
//     if (exists) return res.status(400).json({ message: "User already exists" });

//     const hashed = await bcrypt.hash(password, 10);

//     const user = await User.create({
//       fullName,
//       email,
//       password: hashed,
//       organization,
//       specialty,
//       certificate: req.file.path,
//       status: "Pending"
//     });

//     // send acknowledgment email
//     await sendEmail(email, "Signup Received", 
//       "Your signup request has been received. You will be notified after admin approval."
//     );

//     res.status(201).json({ success: true, message: "Signup request submitted" });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Server error" });
//   }
// };

// export const loginUser = async (req, res) => {
//   try {
//     const { email, password } = req.body;
//     const user = await User.findOne({ email });
//     if (!user) return res.status(400).json({ message: "Invalid credentials" });

//     const isMatch = await bcrypt.compare(password, user.password);
//     if (!isMatch) return res.status(400).json({ message: "Invalid credentials" });

//     if (user.status !== "Approved") {
//       return res.status(403).json({ message: "Your account is not approved yet" });
//     }

//     const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "1d" });

//     res.cookie("userToken", token, {
//       httpOnly: true,
//       secure: process.env.NODE_ENV === "production",
//       sameSite: "strict"
//     });

//     res.json({ success: true, message: "Login successful" });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Server error" });
//   }
// };














// // controllers/userController.js
// import User from "../models/User.js";
// import sendEmail from "../utils/email.js";

// // 👉 Get all users (admin view)
// export const getAllUsers = async (req, res) => {
//   try {
//     const users = await User.find().sort({ createdAt: -1 });
//     res.json(users);
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Server error" });
//   }
// };

// // 👉 Approve user
// export const approveUser = async (req, res) => {
//   try {
//     const user = await User.findById(req.params.id);
//     if (!user) return res.status(404).json({ message: "User not found" });

//     user.status = "Approved";
//     await user.save();

//     await sendEmail(
//       user.email,
//       "Account Approved",
//       `<p>Hi <strong>${user.fullName}</strong>,</p>
//        <p>Your signup request has been <b>approved</b>. You can now log in.</p>`
//     );

//     res.json({ success: true, message: "User approved" });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Server error" });
//   }
// };

// // 👉 Reject user
// export const rejectUser = async (req, res) => {
//   try {
//     const user = await User.findById(req.params.id);
//     if (!user) return res.status(404).json({ message: "User not found" });

//     user.status = "Rejected";
//     await user.save();

//     await sendEmail(
//       user.email,
//       "Account Rejected",
//       `<p>Hi <strong>${user.fullName}</strong>,</p>
//        <p>Unfortunately, your signup request has been <b>rejected</b>. Contact admin for details.</p>`
//     );

//     res.json({ success: true, message: "User rejected" });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Server error" });
//   }
// };














import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import crypto from "crypto";
import User from "../models/User.js";
import sendEmail from "../utils/email.js";


export const signupUser = async (req, res) => {
  try {
    const { fullName, email, password, organization, specialty } = req.body;
    if (!req.file) return res.status(400).json({ message: "Certificate required" });

    // check existing
    const exists = await User.findOne({ email });
    if (exists) return res.status(400).json({ message: "User already exists" });

    const hashed = await bcrypt.hash(password, 10);

    const user = await User.create({
      fullName,
      email,
      password: hashed,
      organization,
      specialty,
      certificate: req.file.path,
      status: "Pending"
    });

    // Send response immediately after user creation
    res.status(201).json({ success: true, message: "Signup request submitted" });

    // Try to send acknowledgment email, but don't fail signup if it fails
    try {
      await sendEmail(
        email,
        "Signup Received",
        "Your signup request has been received. You will be notified after admin approval."
      );
    } catch (emailErr) {
      console.error("Email sending failed:", emailErr);
      // Optionally log this somewhere for admin review
    }
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
};









// // Login user
// export const loginUser = async (req, res) => {
//   try {
//     const { email, password } = req.body;
//     const user = await User.findOne({ email });
//     if (!user) return res.status(400).json({ message: "Invalid credentials" });

//     const isMatch = await bcrypt.compare(password, user.password);
//     if (!isMatch) return res.status(400).json({ message: "Invalid credentials" });

//     if (user.status !== "Approved") {
//       return res.status(403).json({ message: "Your account is not approved yet" });
//     }

//     const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "1d" });

//     res.cookie("userToken", token, {
//       httpOnly: true,
//       secure: process.env.NODE_ENV === "production",
//       sameSite: "strict"
//     });

//     res.json({ success: true, message: "Login successful" });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Server error" });
//   }
// };






export const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ message: "Invalid credentials" });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ message: "Invalid credentials" });

    if (user.status !== "Approved") {
      return res.status(403).json({ message: "Your account is not approved yet" });
    }

    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "1d" });

    res.cookie("userToken", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict"
    });

    // FIX: Return user object in response
    res.json({ success: true, message: "Login successful", user });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
};

export const changePassword = async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;
    const userId = req.user.id;

    // Validate input
    if (!currentPassword || !newPassword) {
      return res.status(400).json({ success: false, message: "Current and new password are required" });
    }

    // Find user with password field included
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ success: false, message: "User not found" });
    }

    // Verify current password
    const isMatch = await bcrypt.compare(currentPassword, user.password);
    if (!isMatch) {
      return res.status(400).json({ success: false, message: "Current password is incorrect" });
    }

    // Hash new password
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    // Update password
    user.password = hashedPassword;
    await user.save();

    res.json({ success: true, message: "Password changed successfully" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

export const forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ success: false, message: "Email is required" });
    }

    // Check if user exists - don't reveal in response for security
    const user = await User.findOne({ email });

    // Generate reset token (raw token to send in email)
    const resetToken = crypto.randomBytes(32).toString("hex");
    
    // Hash token before saving to DB
    const hashedToken = crypto.createHash("sha256").update(resetToken).digest("hex");
    
    // Set expiry time (5 minutes)
    const expiryTime = new Date(Date.now() + 5 * 60 * 1000);

    if (user) {
      // Save hashed token and expiry to DB
      user.resetToken = hashedToken;
      user.resetTokenExpiry = expiryTime;
      await user.save();

      // Send email with reset link
      const resetLink = `${process.env.FRONTEND_URL}/reset-password?token=${resetToken}`;
      const emailContent = `
        <h2>Password Reset Request</h2>
        <p>Hello ${user.fullName},</p>
        <p>We received a request to reset your password. Click the link below to reset it:</p>
        <p><a href="${resetLink}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Reset Password</a></p>
        <p>This link expires in 5 minutes.</p>
        <p>If you didn't request this, please ignore this email.</p>
        <p>Best regards,<br>The Team</p>
      `;

      try {
        await sendEmail(email, "Password Reset Request", emailContent);
      } catch (emailErr) {
        console.error("Email sending failed:", emailErr);
      }
    }

    // Always return same message for security (don't reveal if email exists)
    res.json({
      success: true,
      message: "If an account exists with that email, a password reset link has been sent."
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

// Password validation rules (mirrors frontend PasswordStrength rules)
function validatePassword(password) {
  if (!password || password.length < 8) {
    return "Password must have at least 8 characters.";
  }
  if (!/[A-Z]/.test(password)) {
    return "Password must have at least 1 uppercase letter.";
  }
  if (!/[^A-Za-z0-9]/.test(password)) {
    return "Password must have at least 1 special character.";
  }
  return null;
}

export const resetPassword = async (req, res) => {
  try {
    const { token, newPassword } = req.body;

    if (!token || !newPassword) {
      return res.status(400).json({ success: false, message: "Token and new password are required" });
    }

    // Validate password strength
    const pwdError = validatePassword(newPassword);
    if (pwdError) {
      return res.status(400).json({ success: false, message: pwdError });
    }

    // Hash the token from request to compare with DB
    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

    // Find user with matching token that hasn't expired
    const user = await User.findOne({
      resetToken: hashedToken,
      resetTokenExpiry: { $gt: new Date() }
    });

    if (!user) {
      return res.status(400).json({ success: false, message: "Invalid or expired reset token" });
    }

    // Hash new password
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    // Update password and clear reset token fields
    user.password = hashedPassword;
    user.resetToken = null;
    user.resetTokenExpiry = null;
    await user.save();

    res.json({ success: true, message: "Password reset successfully. You can now log in with your new password." });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: "Server error" });
  }
};