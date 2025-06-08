#ifndef _CANDIDATE_H_
#define _CANDIDATE_H_
#include "PlotSet.h"

class MonoCandidate
{
public:
  MonoCandidate() {}

  MonoCandidate(double sh, double satsh, double dedxsig, double tiso, double xyp0, double xyp1, double xyp2,
                double rzp0, double rzp1, double rzp2,
                double dist, double f51, double f15,
                double e55, double e99, double hiso, double eta,
                double phi, double mono_eta, double mono_phi, double amon_eta, double amon_phi,
                double mono_E, double mono_Et, double amon_E, double amon_Et,
                double event, double NPV, double PFMET_pt, bool matched_to_mono,
                double eRight, double eLeft, double eTop, double eBottom, double eMax,
                double e2x5Right, double e2x5Left, double e2x5Top, double e2x5Bottom, double e2x5Max) :
      subHits_(sh), subSatHits_(satsh), dEdXSig_(dedxsig), tIso_(tiso),
      xyp0_(xyp0), xyp1_(xyp1), xyp2_(xyp2),
      rzp0_(rzp0), rzp1_(rzp1), rzp2_(rzp2),
      dist_(dist), f51_(f51), f15_(f15), e55_(e55), e99_(e99), hIso_(hiso), 
      eta_(eta), phi_(phi), mono_eta_(mono_eta), mono_phi_(mono_phi),
      amon_eta_(amon_eta), amon_phi_(amon_phi),
      mono_E_(mono_E), mono_Et_(mono_Et), amon_E_(amon_E), amon_Et_(amon_Et),
      event_(event), NPV_(NPV), PFMET_pt_(PFMET_pt), matched_to_mono_(matched_to_mono),
      eRight_(eRight), eLeft_(eLeft), eTop_(eTop), eBottom_(eBottom), eMax_(eMax),
      e2x5Right_(e2x5Right), e2x5Left_(e2x5Left), e2x5Top_(e2x5Top), e2x5Bottom_(e2x5Bottom), e2x5Max_(e2x5Max) {}

  MonoCandidate(const MonoCandidate &mc) :
      subHits_(mc.subHits_), subSatHits_(mc.subSatHits_), dEdXSig_(mc.dEdXSig_), tIso_(mc.tIso_),
      xyp0_(mc.xyp0_), xyp1_(mc.xyp1_), xyp2_(mc.xyp2_),
      rzp0_(mc.rzp0_), rzp1_(mc.rzp1_), rzp2_(mc.rzp2_),
      dist_(mc.dist_), f51_(mc.f51_), f15_(mc.f15_), e55_(mc.e55_), e99_(mc.e99_), hIso_(mc.hIso_), 
      eta_(mc.eta_), phi_(mc.phi_), mono_eta_(mc.mono_eta_), mono_phi_(mc.mono_phi_),
      amon_eta_(mc.amon_eta_), amon_phi_(mc.amon_phi_),
      mono_E_(mc.mono_E_), mono_Et_(mc.mono_Et_), amon_E_(mc.amon_E_), amon_Et_(mc.amon_Et_),
      event_(mc.event_), NPV_(mc.NPV_), PFMET_pt_(mc.PFMET_pt_), matched_to_mono_(mc.matched_to_mono_),
      eRight_(mc.eRight_), eLeft_(mc.eLeft_), eTop_(mc.eTop_), eBottom_(mc.eBottom_), eMax_(mc.eMax_),
      e2x5Right_(mc.e2x5Right_), e2x5Left_(mc.e2x5Left_), e2x5Top_(mc.e2x5Top_), e2x5Bottom_(mc.e2x5Bottom_), e2x5Max_(mc.e2x5Max_) {}

  ~MonoCandidate() {}

  bool operator<(const MonoCandidate &mc) const {
    if (dEdXSig_ > mc.dEdXSig_) return true;
    if (dEdXSig_ == mc.dEdXSig_) return f51_ > mc.f51_;
    return false;
  }

  double subHits_;
  double subSatHits_;
  double dEdXSig_;
  double tIso_;
  double xyp0_;
  double xyp1_;
  double xyp2_;
  double rzp0_;
  double rzp1_;
  double rzp2_;
  double dist_;
  double f51_;
  double f15_;
  double e55_;
  double e99_;
  double Cross_;
  double hIso_;
  double eta_;
  double phi_;
  double event_;
  double NPV_;
  double mono_eta_;
  double mono_phi_;
  double amon_eta_;
  double amon_phi_;
  double PFMET_pt_;
  double mono_E_;
  double mono_Et_;
  double amon_E_;
  double amon_Et_;
  bool matched_to_mono_;

  double eRight_;
  double eLeft_;
  double eTop_;
  double eBottom_;
  double eMax_;

  double e2x5Right_;
  double e2x5Left_;
  double e2x5Top_;
  double e2x5Bottom_;
  double e2x5Max_;
};

class Photon
{
public:
  Photon() {}
  Photon(double pho_eta, double pho_phi, double pho_pt)
    : pho_eta_(pho_eta), pho_phi_(pho_phi), pho_pt_(pho_pt) {}
  Photon(const Photon &mc)
    : pho_eta_(mc.pho_eta_), pho_phi_(mc.pho_phi_), pho_pt_(mc.pho_pt_) {}

  double pho_eta_;
  double pho_phi_;
  double pho_pt_;

  ~Photon() {}
};

#endif
