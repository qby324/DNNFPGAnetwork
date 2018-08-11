#! /usr/bin/perl -w

# Sat Nov 23 16:01:12 JST 2013
# shinot
# Makes autoencoder from Kaldi format RBMs.
# Output autoencoder is in Kaldi format.

if (@ARGV < 1) {
    die "$0 rbm1 rbm2 ..."
}

@aff=();
@hbs=();
@vbs=();
for ($i=0; $i<@ARGV; $i++) {
    $rbmf = $ARGV[$i];
    open(IN, $rbmf) || die "$! : $rbmf\n";
    $_ = <IN>; die "Error: \"<Nnet>\" is expected\n" if (!/Nnet/);
    $_ = <IN>; die "Error: \"<rbm> nnum inum\" is expected\n" if (!/<rbm> (\d+) (\d+)/);
    $nnum = $1;
    $inum = $2;
    $_ = <IN>; die "Error: \"type type  [\" nnum inum is expected\n" if (!/\[/);
    for ($j=0; $j<$nnum; $j++) {
	$_ = <IN>;
	s/\]//;
	s/^\s+//;
	@{$aff[$i][$j]} = split;
	if ($inum != @{$aff[$i][$j]}) {
	    die "Error: Inconsistent dim $inum\n";
	}
    }
    $_ = <IN>; s/\s*\[\s*//; s/\s*\]\s*//;
    @{$vbs[$i]} = split;
    $_ = <IN>; s/\s*\[\s*//; s/\s*\]\s*//;
    @{$hbs[$i]} = split;
    $_ = <IN>; die "Error: \"<\/Nnet>\" is expected\n$_" if (!/<\/Nnet>/);
    close(IN);
}

# encoder part
print "<Nnet> \n";
for ($k=0; $k<@aff; $k++) {
    $inum = @{$aff[$k][0]};
    $nnum = @{$aff[$k]};
    print "<affinetransform> $nnum $inum \n";
    print " [\n ";
    for ($i=0; $i<$nnum; $i++) {
	for ($j=0; $j<$inum; $j++) {
	    $val = $aff[$k][$i][$j];
	    print " $val";
	}
	if ($i==$nnum-1) {
	    print " ]\n";
	} else {
	    print " \n ";
	}
    }
    $nhbs = @{$hbs[$k]};
    die "aseert hbs vs nnum $nhbs $nnum\n" if ($nhbs != $nnum);
    print " [";
    for ($i=0; $i<$nnum; $i++) {
	$val = $hbs[$k][$i];
	print " $val";
    }
    print " ] \n";
    print "<sigmoid> $nnum $nnum \n";
}

# decoder part
for ($k=$#aff; 0<=$k; $k--) {
    $nnum = @{$aff[$k][0]}; # aff[k] is a inum x nnum matrix
    $inum = @{$aff[$k]};
    print "<affinetransform> $nnum $inum \n";
    print " [\n";
    for ($i=0; $i<$nnum; $i++) {
	for ($j=0; $j<$inum; $j++) {
	    $val = $aff[$k][$j][$i];
	    print " $val";
	}
	if ($i==$nnum-1) {
	    print " ]\n";
	} else {
	    print " \n ";
	}
    }
    $nvbs = @{$vbs[$k]};
    die "aseert(decode) vbs vs nnum $nvbs $inum\n" if ($nvbs != $nnum);
    print " [";
    for ($i=0; $i<$nnum; $i++) {
	$val = $vbs[$k][$i];
	print " $val";
    }
    print " ] \n";
    if ($k!=0) {
	print "<sigmoid> $nnum $nnum \n";
    }
}

print "</Nnet> \n";
