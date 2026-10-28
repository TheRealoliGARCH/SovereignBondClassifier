(* SovereignBondClassifier.wl
   Wolfram Language implementation of structural bond pricing and a
   three-cluster sovereign-risk classifier. Synthetic example only. *)

BeginPackage["SovereignBondClassifier`"];

ContinuousDiscountFactor::usage = "ContinuousDiscountFactor[t,s,rate] computes Exp[-Integrate_t^s rate[u] du].";
ContinuousBondPrice::usage = "ContinuousBondPrice[t,T,couponFlow,terminalPayment,rate] prices continuous coupons plus a terminal payment.";
ContinuousDuration::usage = "ContinuousDuration[t,T,couponFlow,terminalPayment,rate] computes duration in years.";
EstimateStructuralComponent::usage = "EstimateStructuralComponent[y,z] estimates an OLS intercept/common component and feature coefficients, with a rank check.";
FitSovereignRiskClassifier::usage = "FitSovereignRiskClassifier[data,featureNames] imputes missing values, standardizes features, and fits three K-means clusters.";
MapRiskLabels::usage = "MapRiskLabels[fit,riskScores,defaultEvents] maps clusters to Safe/Risky/Default, requiring default-event evidence in the highest-risk cluster.";
SovereignClassifierDemo::usage = "SovereignClassifierDemo[] runs a synthetic demonstration.";

Begin["`Private`"];

ContinuousDiscountFactor[t_?NumericQ, s_?NumericQ, rate_] := Module[{v},
  If[s < t, Return[Failure["InvalidTime", <|"Message" -> "s must be >= t."|>]]];
  If[s == t, Return[1.]];
  v = Quiet@Check[NIntegrate[rate[u], {u, t, s}], $Failed];
  If[v === $Failed || ! NumericQ[v],
    Failure["IntegrationFailed", <|"Message" -> "Rate integral failed."|>],
    Exp[-v]
  ]
];

ContinuousBondPrice[t_?NumericQ, maturity_?NumericQ, couponFlow_, terminalPayment_?NumericQ, rate_] :=
 Module[{discount, pv},
  If[maturity < t, Return[Failure["InvalidTime", <|"Message" -> "Maturity must be >= t."|>]]];
  If[maturity == t, Return[terminalPayment]];
  discount[x_?NumericQ] := ContinuousDiscountFactor[t, x, rate];
  pv = Quiet@Check[NIntegrate[couponFlow[s] discount[s], {s, t, maturity}], $Failed];
  If[pv === $Failed || ! NumericQ[pv],
    Failure["IntegrationFailed", <|"Message" -> "Coupon present-value integral failed."|>],
    pv + terminalPayment discount[maturity]
  ]
];

ContinuousDuration[t_?NumericQ, maturity_?NumericQ, couponFlow_, terminalPayment_?NumericQ, rate_] :=
 Module[{discount, price, weightedPV},
  If[maturity < t, Return[Failure["InvalidTime", <|"Message" -> "Maturity must be >= t."|>]]];
  If[maturity == t, Return[0.]];
  discount[x_?NumericQ] := ContinuousDiscountFactor[t, x, rate];
  price = Quiet@Check[
    NIntegrate[couponFlow[s] discount[s], {s, t, maturity}] + terminalPayment discount[maturity],
    $Failed
  ];
  If[price === $Failed || ! NumericQ[price] || price <= 0,
    Return[Failure["InvalidPrice", <|"Message" -> "Duration requires a positive finite bond price."|>]]
  ];
  weightedPV = Quiet@Check[
    NIntegrate[(s - t) couponFlow[s] discount[s], {s, t, maturity}] +
      (maturity - t) terminalPayment discount[maturity],
    $Failed
  ];
  If[weightedPV === $Failed || ! NumericQ[weightedPV],
    Failure["IntegrationFailed", <|"Message" -> "Duration integral failed."|>],
    weightedPV/price
  ]
];

EstimateStructuralComponent[y_List, z_?MatrixQ] := Module[
  {yy = N[y], zz = N[z], design, rank, coef, fitted, residuals},
  If[Length[yy] != Length[zz],
    Return[Failure["DimensionMismatch", <|"Message" -> "y and z need the same row count."|>]]
  ];
  If[!VectorQ[yy, NumericQ] || !MatrixQ[zz, NumericQ],
    Return[Failure["NonNumericInput", <|"Message" -> "Inputs must be numeric."|>]]
  ];
  design = Join[Transpose[{ConstantArray[1., Length[yy]]}], zz, 2];
  rank = MatrixRank[design];
  If[rank < Last[Dimensions[design]],
    Return[<|"Identifiable" -> False, "Rank" -> rank,
      "NumberOfParameters" -> Last[Dimensions[design]],
      "Message" -> "The intercept/common component is not separately identified under this design matrix."|>]
  ];
  coef = LeastSquares[design, yy];
  fitted = design.coef;
  residuals = yy - fitted;
  <|"Identifiable" -> True, "CommonComponentIntercept" -> First[coef],
    "FeatureCoefficients" -> Rest[coef], "Coefficients" -> coef,
    "FittedValues" -> fitted, "Residuals" -> residuals,
    "ResidualSumOfSquares" -> Total[residuals^2], "Rank" -> rank|>
];

medianImpute[data_?MatrixQ] := Module[{x = N[data], medians},
  medians = Table[
    With[{v = Select[x[[All, j]], NumericQ[#] && FreeQ[#, Indeterminate | Infinity | -Infinity] &]},
      If[v === {}, 0., Median[v]]
    ],
    {j, Length[First[x]]}
  ];
  x = MapIndexed[
    If[NumericQ[#1] && FreeQ[#1, Indeterminate | Infinity | -Infinity],
      #1, medians[[First[#2]]]
    ] &, x, {2}
  ];
  <|"Data" -> x, "Medians" -> medians|>
];

standardizeColumns[data_?MatrixQ] := Module[{x = N[data], means, scales, out},
  means = Mean /@ Transpose[x];
  scales = StandardDeviation /@ Transpose[x];
  scales = Map[If[!NumericQ[#] || # == 0, 1., #] &, scales];
  out = Table[(x[[i, j]] - means[[j]])/scales[[j]],
    {i, Length[x]}, {j, Length[First[x]]}];
  <|"Data" -> out, "Means" -> means, "Scales" -> scales|>
];

FitSovereignRiskClassifier[data_?MatrixQ, featureNames_List] := Module[
  {n, imputed, standardized, x, clusters, centers, ids},
  n = Length[data];
  If[n < 3, Return[Failure["InsufficientData", <|"Message" -> "Supply at least three observations."|>]]];
  If[Length[First[data]] != Length[featureNames],
    Return[Failure["FeatureNameMismatch", <|"Message" -> "Feature-name count must match data columns."|>]]
  ];
  imputed = medianImpute[data /. Missing[] -> Indeterminate];
  standardized = standardizeColumns[imputed["Data"]];
  x = standardized["Data"];
  clusters = Quiet@Check[FindClusters[x, 3, Method -> "KMeans"], $Failed];
  If[clusters === $Failed || Length[clusters] != 3,
    Return[Failure["ClusteringFailed", <|"Message" -> "K-means did not return exactly three clusters."|>]]
  ];
  centers = Mean /@ clusters;
  ids = Table[First@Ordering[EuclideanDistance[x[[i]], #] & /@ centers, 1], {i, n}];
  <|"FeatureNames" -> featureNames, "OriginalData" -> data,
    "ImputedData" -> imputed["Data"], "ImputationMedians" -> imputed["Medians"],
    "StandardizedData" -> x, "StandardizationMeans" -> standardized["Means"],
    "StandardizationScales" -> standardized["Scales"],
    "ClusterCentersStandardized" -> centers, "ClusterIDs" -> ids,
    "ClusterSizes" -> Counts[ids], "NumberOfClusters" -> 3,
    "Method" -> "KMeans",
    "Warning" -> "Cluster IDs are arbitrary; they are not Safe/Risky/Default labels until explicitly mapped and validated."|>
];

MapRiskLabels[fit_Association, riskScores_List, defaultEvents_List] := Module[
  {ids, n, clusters, means, ordered, highest, mapping, labels, eventsInHighest},
  ids = fit["ClusterIDs"]; n = Length[ids];
  If[Length[riskScores] != n || Length[defaultEvents] != n,
    Return[Failure["DimensionMismatch", <|"Message" -> "riskScores and defaultEvents need one value per observation."|>]]
  ];
  If[!VectorQ[riskScores, NumericQ],
    Return[Failure["InvalidRiskScores", <|"Message" -> "Risk scores must be numeric."|>]]
  ];
  clusters = Sort[DeleteDuplicates[ids]];
  means = Association@Table[c -> Mean[Pick[riskScores, ids, c]], {c, clusters}];
  ordered = SortBy[clusters, means[#] &];
  highest = Last[ordered];
  eventsInHighest = Pick[defaultEvents, ids, highest];
  If[!AnyTrue[eventsInHighest, TrueQ[#] || # === 1 &],
    Return[Failure["NoDefaultEvidence", <|
      "Message" -> "Refusing to label the highest-risk cluster Default: no recorded default event occurs in it.",
      "ClusterRiskScoreMeans" -> means, "HighestRiskCluster" -> highest|>]]
  ];
  mapping = AssociationThread[ordered, {"Safe", "Risky", "Default"}];
  labels = mapping /@ ids;
  Join[fit, <|"RiskScoreMeansByCluster" -> means,
    "ClusterToLabelMapping" -> mapping, "Labels" -> labels,
    "DefaultEventValidation" -> "The Default label is assigned to the highest-risk cluster only because it contains a recorded default event."|>]
];

SovereignClassifierDemo[] := Module[{names, data, risk, events, fit, mapped},
  SeedRandom[20261028];
  names = {"YieldSpread", "DebtToGDP", "Inflation", "FXVolatility"};
  data = Join[
    RandomVariate[NormalDistribution[0.6, 0.25], {12, 4}],
    RandomVariate[NormalDistribution[2.0, 0.45], {12, 4}],
    RandomVariate[NormalDistribution[4.0, 0.65], {12, 4}]
  ];
  risk = data[[All, 1]] + 0.35 data[[All, 2]] + 0.2 data[[All, 3]];
  events = Join[ConstantArray[False, 24], ConstantArray[True, 12]];
  fit = FitSovereignRiskClassifier[data, names];
  If[FailureQ[fit], Return[fit]];
  mapped = MapRiskLabels[fit, risk, events];
  If[FailureQ[mapped], Return[mapped]];
  <|"SyntheticData" -> Dataset[Map[AssociationThread[names, #] &, data]],
    "ClusterSizes" -> mapped["ClusterSizes"],
    "ClusterToLabelMapping" -> mapped["ClusterToLabelMapping"],
    "Labels" -> mapped["Labels"],
    "Caveat" -> "Synthetic demonstration only; not a calibrated sovereign default model."|>
];

End[];
EndPackage[];
