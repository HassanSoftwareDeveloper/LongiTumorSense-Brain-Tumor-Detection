// // routes/adminRoutes.js
// import express from "express";
// import bcrypt from "bcryptjs";
// import jwt from "jsonwebtoken";
// import { adminLogin, adminLogout } from "../controllers/adminAuthController.js";


// const router = express.Router();

// // Temporary admin credentials (later we will fetch from DB)
// // const ADMIN_EMAIL = "admin@gmail.com";
// // const ADMIN_PASSWORD = "admin123";
//  // In DB this should be hashed
//  router.post("/login", adminLogin);
// router.post("/logout", adminLogout);

// // 🔑 Admin Login
// router.post("/login", async (req, res) => {
//   const { email, password } = req.body;

//   try {
//     if (email !== ADMIN_EMAIL || password !== ADMIN_PASSWORD) {
//       return res.status(401).json({ message: "Invalid email or password" });
//     }

//     // Generate JWT
//     const token = jwt.sign({ email }, process.env.JWT_SECRET || "secret", {
//       expiresIn: "1h",
//     });

//     // Set cookie
//     res.cookie("token", token, {
//       httpOnly: true,
//       secure: false, // true in production with HTTPS
//       sameSite: "lax",
//     });

//     return res.json({ message: "Login successful" });
//   } catch (err) {
//     return res.status(500).json({ message: "Server error" });
//   }
// });



// export default router;














// routes/adminRoutes.js
import express from "express";
import User from "../models/User.js";
import { adminLogin, adminLogout } from "../controllers/adminauthController.js";
import sendEmail from "../utils/email.js";


const router = express.Router();

// Use controller-based routes only
router.post("/login", adminLogin);
router.post("/logout", adminLogout);

// For admin (in adminRoutes.js)
router.post("/logout", (req, res) => {
  res.clearCookie("adminToken");
  res.json({ success: true, message: "Logged out" });
});

router.get("/requests", async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (err) {
    res.status(500).json({ message: "Server error" });
  }
});



// Approve user
router.patch("/approve/:id", async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { status: "Approved" },
      { new: true }
    );
    if (!user) return res.status(404).json({ message: "User not found" });

    // Send approval email
    try {
      await sendEmail(
        user.email,
        "Account Approved",
        `Dear ${user.fullName || "User"},\n\nCongratulations! Your account has been approved. You can now log in and use the platform.\n\nBest regards,\nAdmin Team`
      );
      } catch (emailErr) {
      console.error("Approval email sending failed:", emailErr);
    }

    res.json(user);
  } catch (err) {
    res.status(500).json({ message: "Server error" });
  }
});

// Reject user
router.patch("/reject/:id", async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { status: "Rejected" },
      { new: true }
    );
    if (!user) return res.status(404).json({ message: "User not found" });

    // Send rejection email
    try {
      await sendEmail(
        user.email,
        "Account Rejected",
        `Dear ${user.fullName || "User"},\n\nWe regret to inform you that your account request has been rejected. If you believe this is a mistake or need further information, please contact support.\n\nBest regards,\nAdmin Team`
      );
    } catch (emailErr) {
      console.error("Rejection email sending failed:", emailErr);
    }

    res.json(user);
  } catch (err) {
    res.status(500).json({ message: "Server error" });
  }
});

export default router;
