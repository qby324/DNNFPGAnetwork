#! /usr/bin/perl

# Convert MLF to ark iv file

# Sat Jan  3 13:24:35 JST 2015
# shinot

if (@ARGV != 2) {
    die "$0 mlf mapfile\n";
}

$mlff = $ARGV[0];
$mapf = $ARGV[1];

%map = ();
&readmap($mapf);

%ref = ();
&readmlf($mlff);

foreach $id (@mlf) {
    print "$id";
    for ($s=0; $s<@{$mlf{$id}{"start"}}; $s++) {
	$start = $mlf{$id}{"start"}[$s];
	$end = $mlf{$id}{"end"}[$s];
	$word = $mlf{$id}{"word"}[$s];
	if (! defined $map{$word}) {
	    die "Error : $word is not found in mapfile $mapf\n";
	}
	$y = $map{$word};
	for ($i=0; $i<$end-$start; $i++) {
	    print " $y";
	}
    }
    print "\n";
}

# Read map file
# fills %map
sub readmap($) {
    my ($file) = @_;

    open(IN, $file) || die "$! : $file\n";
    while (<IN>) {
	chomp;
	@line = split;
	if (@line != 2) {
	    die "Unexpected format. $_\n";
	}
	($id, $word) = @line;
	$map{$word} = $id;
    }
}

# Read HTK MLF
# fills %mlf
sub readmlf($) {
    my ($file) = @_;
    my ($id, $start, $end, $word);

    open(IN, $file) || die "$! : $file\n";
    while (<IN>) {
	chomp;
	if (/#!MLF!#/) {
	    next;
	} elsif (/^\"(.*\/)*([^\/]+)\.(rec|lab)/) {
	    $id = $2;
	    $mlf[@mlf] = $id;
	    @{$mlf{$id}{"start"}} = ();
	    @{$mlf{$id}{"end"}} = ();
	} elsif (/^(\d+) (\d+) (\S+)/) {
	    $start = $1;
	    $end = $2;
	    $word = $3;
	    if (! $id) {
		die "Unexpected format. ID is not found\n";
	    }
            if (($start+2)%100000 > 4 || ($end+2)%100000 > 4) {
                die "Error (evalmap.pl): Unexpected alignment time unit: $file : $_\n";
            }
            $start = int(($start+2)/100000);
            $end = int(($end+2)/100000);  # In MLF, frame range is [start, end). Frame ID starts with 0.
	    $word = $3;
	    $mlf{$id}{"start"}[@{$mlf{$id}{"start"}}] = $start;
	    $mlf{$id}{"end"}[@{$mlf{$id}{"end"}}] = $end;
	    $mlf{$id}{"word"}[@{$mlf{$id}{"word"}}] = $word;
	} elsif (/^\./) {
	    $id = "";
	} else {
	    die "Error: Unexpected format: $file : $_\n";
	}
    }
}
