# Why Data Quality Matters at Wheel & Way

Wheel & Way runs a bike-share network across the fictional city of Bramford: six docking
stations, a fleet of classic and electric bikes, and a trip log that grows by a few
thousand rows every day. Every trip a rider takes gets written to a table with a trip
id, a start and end station, a start and end time, a bike type, and a rider category
(member or casual). That table looks simple, but it is the raw material for almost
everything the operations team does: pricing, rebalancing bikes between stations, and
reporting ridership to the city.

None of those downstream uses work if the table itself cannot be trusted. A duration
column that sometimes holds minutes and sometimes holds seconds will silently wreck an
average. A station name typed three different ways will make a station look less busy
than it really is, because its trips got split across three labels instead of counted
under one. A handful of trips with an end time earlier than the start time will pull an
average duration negative if nobody notices them first.

This course treats data cleaning as its own skill, separate from the modelling or
reporting that comes after it. The habit we are building is simple to state: before you
trust a number, you inspect the column it came from. That means looking at value counts,
checking types, plotting a histogram of anything numeric, and reading a sample of raw
rows rather than only the aggregated summary.

Across the ten lessons in this course we will work through the Wheel & Way trip log end
to end: spotting the sentinel values a legacy export system used for 'no data', fixing
columns that were loaded as the wrong type, finding duplicate and near-duplicate trips,
flagging outlier durations, filling in gaps sensibly, cleaning up free-text fields,
parsing timestamps into usable features, and finally validating and exporting a table
the rest of the company can build on.

A theme that will come back again and again is that cleaning is not a single pass you
run once and forget. Every fix you apply should be something you could explain to a
colleague and defend with a number: how many rows changed, what rule you used, and why
that rule was reasonable for this particular column. That discipline is what turns 'I
cleaned the data' into something someone else can check and trust.

By the end of this lesson you should be able to describe, in your own words, at least
three ways a messy trip log could quietly mislead an analyst who never looked past the
summary statistics. Keep that list in mind; the next nine lessons each hand you a
concrete tool for catching one of those failure modes before it reaches a report.

## 1. Practice Questions

### Q1. (Multiple Choice)
A teammate says the average trip duration at Wheel & Way "looks about right, so the data must be clean." What is the strongest reason to be cautious about that conclusion?

- A) Averages are always the wrong statistic for trip durations. - **Incorrect.** Averages are a fine starting point; the issue is trusting them without inspection.
- B) Bike-share companies never have data quality problems. - **Incorrect.** Every operational data source accumulates quality problems over time.
- C) A plausible-looking average can still hide compensating errors, such as some durations recorded in seconds and others in minutes cancelling out in the mean. - **Correct answer.** This is exactly the kind of silent error a summary statistic can mask.
- D) The average should be replaced by the median in every situation. - **Incorrect.** The median has its own uses, but switching statistics does not by itself catch mixed units.

- **Answer:** C

### Q2. (True/False)
Inspecting raw sample rows in addition to summary statistics is a useful habit when starting to clean an unfamiliar dataset.

- **Answer:** True
- **Hint:** The lesson argues that aggregated numbers alone can hide row-level problems.

### Q3. (Open-Ended)
Name two different ways a station name field with inconsistent spelling could distort a report on Wheel & Way's busiest stations, and explain briefly why each matters.

- **Answer:** It can split one station's trips across several labels so none of the variants looks busy on its own, and it can make station-level joins with a lookup table silently drop rows whose spelling does not match any known key.
