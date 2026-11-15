import React from 'react';
import Hero from '../../sections/hero/Hero';
import HowItWorks from '../../sections/howitworks/HowItWorks';
import MissionCTA from '../../sections/mission/MissionCTA';
import Features from '../../sections/whyus/Features';
import Comparison from '../../sections/comparison/Comparison';

function LandingPage() {
  return (
    <>
      <Hero />
      <HowItWorks />
      <Comparison />
      <Features />
      <MissionCTA />
    </>
  );
}

export default LandingPage;
