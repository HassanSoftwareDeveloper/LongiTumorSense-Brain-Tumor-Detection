import mongoose from 'mongoose';


const ReportSchema = new mongoose.Schema({
patient: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true },
uploadedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'Doctor' },
filename: { type: String },
result: { type: Object },
createdAt: { type: Date, default: Date.now }
});


export default mongoose.model('Report', ReportSchema);