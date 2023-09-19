import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { IonicModule } from '@ionic/angular';

import { ExpertPagePageRoutingModule } from './expert-page-routing.module';

import { ExpertPagePage } from './expert-page.page';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    ExpertPagePageRoutingModule
  ],
  declarations: [ExpertPagePage]
})
export class ExpertPagePageModule {}
