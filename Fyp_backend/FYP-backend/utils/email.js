// // import nodemailer from "nodemailer";

// // const transporter = nodemailer.createTransport({
// //   service: "gmail",
// //   auth: {
// //     user: process.env.EMAIL_USER,
// //     pass: process.env.EMAIL_PASS
// //   }
// // });

// // export default async function sendEmail(to, subject, text) {
// //   await transporter.sendMail({
// //     from: process.env.EMAIL_USER,
// //     to,
// //     subject,
// //     text
// //   });
// // }





// // utils/emails.js
// import nodemailer from "nodemailer";

// const transporter = nodemailer.createTransport({
//   service: "gmail",
//   auth: {
//     user: process.env.EMAIL_USER,
//     pass: process.env.EMAIL_PASS,
//   },
// });

// export default async function sendEmail(to, subject, content) {
//   await transporter.sendMail({
//     from: process.env.EMAIL_USER,
//     to,
//     subject,
//     html: content, // <-- supports HTML
//   });
// }




import nodemailer from "nodemailer";
import dotenv from "dotenv";
dotenv.config();

const transporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  port: 587,
  secure: false, // true for port 465
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

export default async function sendEmail(to, subject, content) {
  await transporter.sendMail({
    from: process.env.EMAIL_USER,
    to,
    subject,
    html: content,
  });
}
