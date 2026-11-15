import express from "express";
import { signupUser, loginUser, changePassword, forgotPassword, resetPassword } from "../controllers/authController.js";
import { protect } from "../middleware/authentication.js";
import multer from "multer";

const router = express.Router();

// file upload setup (store locally for now)
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname)
});
const upload = multer({ storage });

router.post("/signup", upload.single("certificate"), signupUser);
  

router.post("/login", loginUser);

router.post("/change-password", protect, changePassword);

router.post("/forgot-password", forgotPassword);

router.post("/reset-password", resetPassword);

router.post("/logout", (req, res) => {
  res.clearCookie("userToken");
  res.json({ success: true, message: "Logged out" });
});



export default router;
