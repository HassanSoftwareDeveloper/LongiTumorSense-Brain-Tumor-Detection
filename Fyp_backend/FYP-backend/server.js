import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import cookieParser from 'cookie-parser';



dotenv.config();

import patientRoutes from './routes/patientRoutes.js';
import mriRoutes from './routes/mriRoutes.js';
import adminRoutes from './routes/adminRoutes.js'; // <-- we'll add this soon
import authRoutes from "./routes/authRoutes.js";
import reportRoutes from "./routes/reportRoutes.js";





const app = express();

// ✅ Configure CORS properly for cookies
app.use(
  cors({
    origin: "http://localhost:5173", // frontend URL
    credentials: true, // allow cookies
  })
);

app.use(cookieParser());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use("/api/reports", reportRoutes);


// connect mongo
// mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
mongoose.connect(process.env.MONGO_URI)
.then(() => console.log('Mongo connected'))
.catch((err) => console.error('Mongo connect err', err));

app.use('/api/patients', patientRoutes);
app.use('/api/mri', mriRoutes);
app.use('/api/admin', adminRoutes); // <-- for login/logout
app.use("/api/users", authRoutes);
// In your backend app.js or server.js
app.use("/uploads", express.static("uploads"));





app.get('/', (req, res) => res.send('Backend running'));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
