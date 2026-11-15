// seedAdmin.js
import mongoose from "mongoose";
import dotenv from "dotenv";
import Admin from "../Backend/models/Admin.js";

dotenv.config();

mongoose.connect(process.env.MONGO_URI).then(async () => {
  const admin = new Admin({
    email: "admin@gmail.com",
    password: "admin123",
  });
  await admin.save();
  console.log("Admin created:", admin.email);
  process.exit();
});
