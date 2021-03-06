Feature: Filtering time series and lists of intervals 
  
  Experimental input comes in the form of time series (or lists of intervals) where each time
  denotes a transition event from open to shut or shut to open. These series should be filtered so
  that only events lasting longuer than a given resolution tau are kept. Events shorter than tau are
  deemed unetected and are merged into the previous segment. The following tests both the bindings
  form c++ to python and the correctness of the implementation.

  The tests are perfomed by engineering random time series such that the filtered result is known
  from the outset.

  Scenario: Filter random time series
    Given 100 random time series with tau=1
     When each time series is filtered
     Then expected filtered time series is obtained

  Scenario: Filter many random time series simultaneously
    Given 100 random time series with tau=1
     When all time series are filtered simultaneously
     Then expected filtered time series is obtained
  
  Scenario: Filter random time intervals
    Given 100 random time intervals with tau=1
     When each list of time intervals is filtered
     Then expected list of filtered time intervals is obtained

  Scenario: Filter random time intervals simultaneously
    Given 100 random time intervals with tau=1
     When all lists of time intervals are filtered simultaneously
     Then expected list of filtered time intervals is obtained
