package org.apache.commons.statistics.distribution;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class InverseBinomialCdfNormalApproximationBiasTest {

    private static final SplittableRandom REPEATABLE_RANDOM = new SplittableRandom(0);

    private static final double PROB_OF_SUCCESS = 0.9;

    private static final double TRIALS_EXP_BASE = 1.01;
    private static final int MIN_TRIALS = 100_000;
    private static final int MAX_TRIALS = 1_000_000_000;

    private static final double MIN_P = 0.8;
    private static final double MAX_P = 0.999;
    private static final int P_STEPS = 20;

    private static final double P_EXP_BASE = 1 / Math.pow((1 - MIN_P) / (1 - MAX_P), 1.0 / P_STEPS);

    public static void main(String[] args) {
        for (double pComplement = 1.0 - MIN_P; pComplement >= 1.0 - MAX_P;
             pComplement *= P_EXP_BASE) {
            double p = 1.0 - pComplement;

            double meanError = estimateMeanError(p);

            System.out.println(p + " " + meanError);
        }
    }

    private static double estimateMeanError(double p) {
        List<Integer> trialsList = new ArrayList<>();
        for (int baseTrials = MIN_TRIALS; baseTrials < MAX_TRIALS;
             baseTrials *= TRIALS_EXP_BASE) {
            int nextTrials = (int) (baseTrials * TRIALS_EXP_BASE);
            // Sample size of 1000 per "trials log" ensures approximately 3-digit precision
            // (+/- 0.001) of MAE. That could be verified by removing seed from REPEATABLE_RANDOM
            // and seeing how this program prints different values from run to run.
            int trialsSampleSize = 1000;
            Collection<Integer> trialsSelection =
                    generateRandomSelection(trialsSampleSize, baseTrials, nextTrials);
            for (Integer trials : trialsSelection) {
                trialsList.add(trials);
            }
        }
        return trialsList
                .stream()
                .parallel()
                .mapToDouble(trials -> {
                    int binomialResult = new BinomialDistribution(trials, PROB_OF_SUCCESS)
                            .inverseCumulativeProbability(p);
                    double approxNormalResult =
                            inverseBinomialCdfNormalApproximation(PROB_OF_SUCCESS, p, trials);
                    return approxNormalResult - binomialResult;
                })
                .average()
                .getAsDouble();
    }

    private static Collection<Integer> generateRandomSelection(
            int sampleSize, int origin, int bound) {
        if (bound - origin <= sampleSize) {
            return IntStream.range(origin, bound).mapToObj(x -> x).collect(Collectors.toList());
        }
        Set<Integer> selected = new HashSet<>();
        while (selected.size() < sampleSize) {
            selected.add(REPEATABLE_RANDOM.nextInt(origin, bound));
        }
        return selected;
    }

    private static double inverseBinomialCdfNormalApproximation(
            double probOfSuccess, double p, int trials) {
        double mean = trials * probOfSuccess;
        double stdDev = Math.sqrt(trials * probOfSuccess * (1 - probOfSuccess));
        return new NormalDistribution(mean, stdDev).inverseCumulativeProbability(p);
    }
}
